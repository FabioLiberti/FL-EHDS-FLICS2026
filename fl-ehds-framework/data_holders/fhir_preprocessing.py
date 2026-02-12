"""
FHIR Preprocessing Module
=========================
Healthcare data preprocessing for FHIR R4 resources.
Addresses the 34% FHIR compliance gap in European healthcare.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import structlog

from core.models import DataCategory
from core.exceptions import FHIRPreprocessingError, FHIRValidationError

logger = structlog.get_logger(__name__)


@dataclass
class FHIRResourceMapping:
    """Mapping from FHIR resource to feature extraction."""

    resource_type: str
    fields: List[str]
    transform: Optional[str] = None
    required: bool = False


class FHIRValidator:
    """
    Validates FHIR R4 resources for FL training compatibility.
    """

    # Required fields for common FHIR resources
    REQUIRED_FIELDS = {
        "Patient": ["id", "birthDate"],
        "Observation": ["id", "code", "valueQuantity"],
        "Condition": ["id", "code", "clinicalStatus"],
        "Procedure": ["id", "code", "status"],
        "MedicationRequest": ["id", "medicationCodeableConcept", "status"],
        "DiagnosticReport": ["id", "code", "status"],
    }

    def __init__(
        self,
        strict: bool = False,
        fhir_version: str = "R4",
    ):
        """
        Initialize FHIR validator.

        Args:
            strict: Enforce strict validation.
            fhir_version: FHIR version to validate against.
        """
        self.strict = strict
        self.fhir_version = fhir_version

    def validate(
        self,
        resource: Dict[str, Any],
    ) -> Tuple[bool, List[str]]:
        """
        Validate a FHIR resource.

        Args:
            resource: FHIR resource dictionary.

        Returns:
            Tuple of (is_valid, list_of_errors).
        """
        errors = []

        # Check resourceType
        if "resourceType" not in resource:
            errors.append("Missing resourceType")
            return False, errors

        resource_type = resource["resourceType"]

        # Check required fields
        if resource_type in self.REQUIRED_FIELDS:
            for field in self.REQUIRED_FIELDS[resource_type]:
                if field not in resource:
                    errors.append(f"Missing required field: {field}")

        # Check id format
        if "id" in resource:
            if not isinstance(resource["id"], str) or len(resource["id"]) == 0:
                errors.append("Invalid resource id")

        # Additional strict validations
        if self.strict:
            errors.extend(self._strict_validate(resource))

        is_valid = len(errors) == 0

        if not is_valid:
            logger.warning(
                "FHIR validation failed",
                resource_type=resource_type,
                errors=errors,
            )

        return is_valid, errors

    def _strict_validate(self, resource: Dict[str, Any]) -> List[str]:
        """Perform strict validation checks."""
        errors = []

        # Check for proper coding systems
        if "code" in resource:
            code = resource["code"]
            if isinstance(code, dict) and "coding" in code:
                for coding in code["coding"]:
                    if "system" not in coding:
                        errors.append("Code missing system URI")

        return errors

    def validate_bundle(
        self,
        bundle: Dict[str, Any],
    ) -> Tuple[bool, Dict[str, List[str]]]:
        """
        Validate a FHIR Bundle.

        Args:
            bundle: FHIR Bundle resource.

        Returns:
            Tuple of (all_valid, resource_errors_dict).
        """
        if bundle.get("resourceType") != "Bundle":
            return False, {"bundle": ["Not a valid Bundle"]}

        errors = {}
        all_valid = True

        entries = bundle.get("entry", [])
        for i, entry in enumerate(entries):
            resource = entry.get("resource", {})
            is_valid, resource_errors = self.validate(resource)
            if not is_valid:
                all_valid = False
                errors[f"entry_{i}"] = resource_errors

        return all_valid, errors


class FHIRPreprocessor:
    """
    Preprocesses FHIR resources for FL training.

    Transforms heterogeneous healthcare data into standardized
    feature representations suitable for machine learning.
    """

    # Default resource mappings for feature extraction
    DEFAULT_MAPPINGS = {
        "Patient": FHIRResourceMapping(
            resource_type="Patient",
            fields=["birthDate", "gender", "deceasedBoolean"],
            transform="demographics",
        ),
        "Observation": FHIRResourceMapping(
            resource_type="Observation",
            fields=["code", "valueQuantity", "effectiveDateTime"],
            transform="numerical",
        ),
        "Condition": FHIRResourceMapping(
            resource_type="Condition",
            fields=["code", "clinicalStatus", "onsetDateTime"],
            transform="categorical",
        ),
        "Procedure": FHIRResourceMapping(
            resource_type="Procedure",
            fields=["code", "status", "performedDateTime"],
            transform="categorical",
        ),
        "MedicationRequest": FHIRResourceMapping(
            resource_type="MedicationRequest",
            fields=["medicationCodeableConcept", "status", "authoredOn"],
            transform="categorical",
        ),
    }

    def __init__(
        self,
        validator: Optional[FHIRValidator] = None,
        normalize: bool = True,
        handle_missing: str = "impute",
        imputation_method: str = "median",
        categorical_encoding: str = "onehot",
        numerical_scaling: str = "standard",
        custom_mappings: Optional[Dict[str, FHIRResourceMapping]] = None,
        strict_validation: bool = False,
    ):
        """
        Initialize FHIR preprocessor.

        Args:
            validator: FHIR validator instance.
            normalize: Apply normalization.
            handle_missing: Missing value strategy ('impute', 'drop', 'error').
            imputation_method: Imputation method ('mean', 'median', 'mode').
            categorical_encoding: Categorical encoding ('onehot', 'label').
            numerical_scaling: Numerical scaling ('standard', 'minmax').
            custom_mappings: Custom resource mappings.
        """
        self.strict_validation = strict_validation
        self.validator = validator or FHIRValidator(strict=strict_validation)
        self.normalize = normalize
        self.handle_missing = handle_missing
        self.imputation_method = imputation_method
        self.categorical_encoding = categorical_encoding
        self.numerical_scaling = numerical_scaling

        self.mappings = {**self.DEFAULT_MAPPINGS}
        if custom_mappings:
            self.mappings.update(custom_mappings)

        # Statistics for scaling (learned from data)
        self._feature_stats: Dict[str, Dict[str, float]] = {}
        self._category_maps: Dict[str, Dict[str, int]] = {}

    def preprocess(
        self,
        resources: List[Dict[str, Any]],
        validate: bool = True,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Preprocess FHIR resources into feature matrix.

        Args:
            resources: List of FHIR resources.
            validate: Whether to validate resources first.

        Returns:
            Tuple of (feature_matrix, feature_names).
        """
        if validate:
            resources = self._validate_and_filter(resources)

        if not resources:
            raise FHIRPreprocessingError("No valid resources to preprocess")

        # Group by resource type
        grouped = self._group_by_type(resources)

        # Extract features from each group
        all_features = []
        all_names = []

        for resource_type, type_resources in grouped.items():
            if resource_type in self.mappings:
                features, names = self._extract_features(
                    type_resources, self.mappings[resource_type]
                )
                all_features.append(features)
                all_names.extend(names)

        if not all_features:
            raise FHIRPreprocessingError("No features extracted from resources")

        # Combine features
        combined = np.hstack(all_features) if len(all_features) > 1 else all_features[0]

        # Apply normalization
        if self.normalize:
            combined = self._normalize_features(combined, all_names)

        logger.info(
            "FHIR preprocessing complete",
            num_resources=len(resources),
            num_features=len(all_names),
            shape=combined.shape,
        )

        return combined, all_names

    def _validate_and_filter(
        self,
        resources: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Validate resources and filter invalid ones."""
        valid = []
        invalid_count = 0

        for resource in resources:
            is_valid, errors = self.validator.validate(resource)
            if is_valid:
                valid.append(resource)
            else:
                invalid_count += 1
                if self.handle_missing == "error":
                    raise FHIRValidationError(
                        resource.get("resourceType", "Unknown"),
                        resource.get("id", "Unknown"),
                        errors,
                    )

        if invalid_count > 0:
            logger.warning(
                "Invalid resources filtered",
                invalid_count=invalid_count,
                valid_count=len(valid),
            )

        return valid

    def _group_by_type(
        self,
        resources: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group resources by resourceType."""
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for resource in resources:
            resource_type = resource.get("resourceType", "Unknown")
            if resource_type not in grouped:
                grouped[resource_type] = []
            grouped[resource_type].append(resource)
        return grouped

    def _extract_features(
        self,
        resources: List[Dict[str, Any]],
        mapping: FHIRResourceMapping,
    ) -> Tuple[np.ndarray, List[str]]:
        """Extract features from resources according to mapping."""
        features = []
        feature_names = []

        for resource in resources:
            resource_features = []

            for field in mapping.fields:
                value = self._extract_field(resource, field)

                if mapping.transform == "numerical":
                    feat_value, feat_name = self._transform_numerical(
                        value, f"{mapping.resource_type}_{field}"
                    )
                elif mapping.transform == "categorical":
                    feat_value, feat_name = self._transform_categorical(
                        value, f"{mapping.resource_type}_{field}"
                    )
                elif mapping.transform == "demographics":
                    feat_value, feat_name = self._transform_demographics(
                        value, field
                    )
                else:
                    feat_value, feat_name = [value], [f"{mapping.resource_type}_{field}"]

                resource_features.extend(feat_value)
                if not feature_names:  # Only set on first resource
                    feature_names.extend(feat_name)

            features.append(resource_features)

        return np.array(features, dtype=np.float32), feature_names

    def _extract_field(
        self,
        resource: Dict[str, Any],
        field: str,
    ) -> Any:
        """Extract a field value from resource."""
        # Handle nested paths (e.g., "valueQuantity.value")
        parts = field.split(".")
        value = resource

        for part in parts:
            if isinstance(value, dict):
                value = value.get(part)
            else:
                value = None
                break

        return value

    def _transform_numerical(
        self,
        value: Any,
        name: str,
    ) -> Tuple[List[float], List[str]]:
        """Transform numerical value."""
        if value is None:
            return [self._impute_numerical(name)], [name]

        if isinstance(value, dict):
            # Handle FHIR Quantity type
            if "value" in value:
                return [float(value["value"])], [name]

        try:
            return [float(value)], [name]
        except (ValueError, TypeError):
            return [self._impute_numerical(name)], [name]

    def _transform_categorical(
        self,
        value: Any,
        name: str,
    ) -> Tuple[List[float], List[str]]:
        """Transform categorical value to encoded representation."""
        if value is None:
            value = "unknown"

        # Handle CodeableConcept
        if isinstance(value, dict):
            if "coding" in value and value["coding"]:
                value = value["coding"][0].get("code", "unknown")
            elif "text" in value:
                value = value["text"]
            else:
                value = "unknown"

        str_value = str(value)

        # Build category map
        if name not in self._category_maps:
            self._category_maps[name] = {}

        if str_value not in self._category_maps[name]:
            self._category_maps[name][str_value] = len(self._category_maps[name])

        if self.categorical_encoding == "label":
            return [float(self._category_maps[name][str_value])], [name]
        else:
            # One-hot encoding placeholder (simplified)
            return [float(self._category_maps[name][str_value])], [name]

    def _transform_demographics(
        self,
        value: Any,
        field: str,
    ) -> Tuple[List[float], List[str]]:
        """Transform demographic fields."""
        if field == "birthDate" and value:
            # Calculate age
            try:
                birth = datetime.strptime(value[:10], "%Y-%m-%d")
                age = (datetime.now() - birth).days / 365.25
                return [age], ["age"]
            except ValueError:
                return [0.0], ["age"]

        elif field == "gender":
            gender_map = {"male": 0, "female": 1, "other": 2, "unknown": 3}
            return [float(gender_map.get(str(value).lower(), 3))], ["gender"]

        elif field == "deceasedBoolean":
            return [1.0 if value else 0.0], ["deceased"]

        return [0.0], [field]

    def _impute_numerical(self, name: str) -> float:
        """Impute missing numerical value."""
        if name in self._feature_stats:
            if self.imputation_method == "mean":
                return self._feature_stats[name].get("mean", 0.0)
            elif self.imputation_method == "median":
                return self._feature_stats[name].get("median", 0.0)
        return 0.0

    def _normalize_features(
        self,
        features: np.ndarray,
        names: List[str],
    ) -> np.ndarray:
        """Normalize feature matrix."""
        normalized = features.copy()

        for i, name in enumerate(names):
            col = features[:, i]

            if self.numerical_scaling == "standard":
                mean = np.mean(col)
                std = np.std(col)
                if std > 0:
                    normalized[:, i] = (col - mean) / std
                self._feature_stats[name] = {"mean": mean, "std": std}

            elif self.numerical_scaling == "minmax":
                min_val = np.min(col)
                max_val = np.max(col)
                if max_val > min_val:
                    normalized[:, i] = (col - min_val) / (max_val - min_val)
                self._feature_stats[name] = {"min": min_val, "max": max_val}

        return normalized

    def get_feature_stats(self) -> Dict[str, Dict[str, float]]:
        """Get learned feature statistics."""
        return self._feature_stats.copy()

    def process_resource(self, resource: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single FHIR resource into feature dictionary.

        Args:
            resource: FHIR resource dictionary.

        Returns:
            Dictionary of extracted features.
        """
        resource_type = resource.get("resourceType", "Unknown")

        # Validate if strict
        if self.strict_validation:
            is_valid, errors = self.validator.validate(resource)
            if not is_valid:
                raise FHIRValidationError(resource_type, resource.get("id", ""), errors)
            # Reject unknown resource types in strict mode
            supported_types = set(self.mappings.keys())
            if resource_type not in supported_types:
                raise FHIRValidationError(
                    resource_type, resource.get("id", ""),
                    [f"Unsupported resource type: {resource_type}"],
                )

        features = {}

        if resource_type == "Patient":
            if "birthDate" in resource:
                try:
                    birth = datetime.strptime(resource["birthDate"][:10], "%Y-%m-%d")
                    features["age"] = (datetime.now() - birth).days / 365.25
                    features["birth_year"] = birth.year
                except ValueError:
                    features["age"] = 0.0
            if "gender" in resource:
                features["gender"] = resource["gender"]
            if "address" in resource and resource["address"]:
                addr = resource["address"][0] if isinstance(resource["address"], list) else resource["address"]
                if isinstance(addr, dict) and "country" in addr:
                    features["country"] = addr["country"]

        elif resource_type == "Observation":
            if "code" in resource:
                code = resource["code"]
                if isinstance(code, dict) and "coding" in code and code["coding"]:
                    features["code"] = code["coding"][0].get("code", "")
            if "valueQuantity" in resource:
                vq = resource["valueQuantity"]
                if isinstance(vq, dict):
                    features["value"] = vq.get("value", 0)
                    features["unit"] = vq.get("unit", "")

        elif resource_type in self.mappings:
            mapping = self.mappings[resource_type]
            for field in mapping.fields:
                value = self._extract_field(resource, field)
                if value is not None:
                    features[field] = value

        if not features:
            features["id"] = resource.get("id", "unknown")

        return features

    def process_batch(self, resources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process a batch of FHIR resources.

        Args:
            resources: List of FHIR resource dictionaries.

        Returns:
            List of feature dictionaries.
        """
        results = []
        for resource in resources:
            try:
                features = self.process_resource(resource)
                results.append(features)
            except FHIRValidationError:
                if self.handle_missing == "error":
                    raise
        return results


class FeatureExtractor:
    """
    Extracts and encodes features from processed records.
    """

    def __init__(
        self,
        feature_columns: List[str],
        categorical_columns: Optional[List[str]] = None,
        missing_value_strategy: str = "default",
        default_values: Optional[Dict[str, Any]] = None,
        encoding_strategy: str = "label",
    ):
        self.feature_columns = feature_columns
        self.categorical_columns = categorical_columns or []
        self.missing_value_strategy = missing_value_strategy
        self.default_values = default_values or {}
        self.encoding_strategy = encoding_strategy
        self._encoders: Dict[str, Dict[str, int]] = {}

    def fit(self, records: List[Dict[str, Any]]) -> "FeatureExtractor":
        """Learn encodings from records."""
        for col in self.categorical_columns:
            unique_values = set()
            for record in records:
                val = record.get(col)
                if val is not None:
                    unique_values.add(str(val))
            self._encoders[col] = {v: i for i, v in enumerate(sorted(unique_values))}
        return self

    def extract(self, record: Dict[str, Any]) -> List[float]:
        """Extract features from a single record."""
        features = []
        for col in self.feature_columns:
            val = record.get(col)

            if val is None:
                val = self.default_values.get(col, 0)

            if col in self.categorical_columns:
                encoder = self._encoders.get(col, {})
                if self.encoding_strategy == "onehot" and encoder:
                    onehot = [0.0] * len(encoder)
                    idx = encoder.get(str(val))
                    if idx is not None:
                        onehot[idx] = 1.0
                    features.extend(onehot)
                else:
                    features.append(float(encoder.get(str(val), 0)))
            else:
                try:
                    features.append(float(val))
                except (ValueError, TypeError):
                    features.append(0.0)

        return features


class DataNormalizer:
    """
    Normalizes feature matrices using standard or min-max scaling.
    """

    def __init__(self, method: str = "standard"):
        self.method = method
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._min: Optional[np.ndarray] = None
        self._max: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray) -> "DataNormalizer":
        """Learn normalization parameters from data."""
        if self.method == "standard":
            self._mean = np.mean(data, axis=0)
            self._std = np.std(data, axis=0)
            self._std[self._std == 0] = 1.0  # prevent division by zero
        elif self.method == "minmax":
            self._min = np.min(data, axis=0)
            self._max = np.max(data, axis=0)
            rng = self._max - self._min
            rng[rng == 0] = 1.0
            self._max = self._min + rng  # adjusted to prevent div/0
        return self

    def transform(self, data: np.ndarray) -> np.ndarray:
        """Normalize data using fitted parameters."""
        if self.method == "standard":
            return (data - self._mean) / self._std
        elif self.method == "minmax":
            return (data - self._min) / (self._max - self._min)
        return data

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        """Reverse normalization."""
        if self.method == "standard":
            return data * self._std + self._mean
        elif self.method == "minmax":
            return data * (self._max - self._min) + self._min
        return data
