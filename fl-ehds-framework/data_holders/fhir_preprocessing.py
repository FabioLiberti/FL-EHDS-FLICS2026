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
        self.validator = validator or FHIRValidator()
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
