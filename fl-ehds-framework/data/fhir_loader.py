#!/usr/bin/env python3
"""
FL-EHDS FHIR R4 Data Loader

Provides functionality to:
1. Load FHIR R4 bundles from files or FHIR servers
2. Extract features from FHIR resources (Patient, Observation, Condition, etc.)
3. Apply opt-out filtering (EHDS Article 71)
4. Convert to FL-ready tensors with proper normalization

Supports both:
- Synthetic FHIR data generation for testing
- Real FHIR server connections for production

Author: Fabio Liberti
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np

# Optional FHIR library
try:
    from fhir.resources.bundle import Bundle
    from fhir.resources.patient import Patient
    from fhir.resources.observation import Observation
    from fhir.resources.condition import Condition
    from fhir.resources.medicationstatement import MedicationStatement
    FHIR_AVAILABLE = True
except ImportError:
    FHIR_AVAILABLE = False
    Bundle = Patient = Observation = Condition = MedicationStatement = None

# Optional requests for FHIR server
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class PatientRecord:
    """Standardized patient record extracted from FHIR resources."""
    patient_id: str
    pseudonymized_id: str

    # Demographics
    age: Optional[float] = None
    gender: Optional[str] = None
    birth_date: Optional[date] = None

    # Clinical features (commonly used)
    bmi: Optional[float] = None
    systolic_bp: Optional[float] = None
    diastolic_bp: Optional[float] = None
    heart_rate: Optional[float] = None
    glucose: Optional[float] = None
    cholesterol: Optional[float] = None
    hemoglobin: Optional[float] = None
    creatinine: Optional[float] = None

    # Diagnosis codes (ICD-10)
    conditions: List[str] = field(default_factory=list)

    # Medications (ATC codes)
    medications: List[str] = field(default_factory=list)

    # Labels/Outcomes
    outcome_30day_mortality: Optional[bool] = None
    outcome_readmission: Optional[bool] = None
    outcome_icu_admission: Optional[bool] = None

    # Metadata
    record_date: Optional[datetime] = None
    source_hospital: Optional[str] = None

    def to_feature_vector(self, feature_spec: List[str]) -> np.ndarray:
        """Convert record to feature vector based on specification."""
        features = []

        feature_map = {
            'age': self.age,
            'gender': 1.0 if self.gender == 'male' else 0.0 if self.gender == 'female' else 0.5,
            'bmi': self.bmi,
            'systolic_bp': self.systolic_bp,
            'diastolic_bp': self.diastolic_bp,
            'heart_rate': self.heart_rate,
            'glucose': self.glucose,
            'cholesterol': self.cholesterol,
            'hemoglobin': self.hemoglobin,
            'creatinine': self.creatinine,
            'num_conditions': len(self.conditions),
            'num_medications': len(self.medications),
        }

        for feat in feature_spec:
            value = feature_map.get(feat, 0.0)
            features.append(value if value is not None else 0.0)

        return np.array(features, dtype=np.float32)


@dataclass
class FLDataset:
    """FL-ready dataset with features and labels."""
    X: np.ndarray  # Feature matrix (n_samples, n_features)
    y: np.ndarray  # Labels (n_samples,)
    patient_ids: List[str]  # Pseudonymized IDs
    feature_names: List[str]
    label_name: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_samples(self) -> int:
        return len(self.y)

    @property
    def n_features(self) -> int:
        return self.X.shape[1]

    def normalize(self, method: str = 'standard') -> 'FLDataset':
        """Normalize features."""
        if method == 'standard':
            mean = self.X.mean(axis=0)
            std = self.X.std(axis=0) + 1e-8
            X_norm = (self.X - mean) / std
        elif method == 'minmax':
            min_val = self.X.min(axis=0)
            max_val = self.X.max(axis=0)
            X_norm = (self.X - min_val) / (max_val - min_val + 1e-8)
        else:
            X_norm = self.X

        return FLDataset(
            X=X_norm,
            y=self.y,
            patient_ids=self.patient_ids,
            feature_names=self.feature_names,
            label_name=self.label_name,
            metadata={**self.metadata, 'normalization': method}
        )


# =============================================================================
# OPT-OUT REGISTRY
# =============================================================================

class OptOutRegistry:
    """
    Manages opt-out status for EHDS Article 71 compliance.

    In production, this would connect to national opt-out registries.
    This implementation provides a local simulation.
    """

    def __init__(self, registry_file: Optional[str] = None):
        self.opted_out: Dict[str, Dict] = {}

        if registry_file and Path(registry_file).exists():
            self.load_registry(registry_file)

    def load_registry(self, filepath: str):
        """Load opt-out records from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        for record in data.get('opt_outs', []):
            citizen_id = record['citizen_id']
            self.opted_out[citizen_id] = {
                'purposes': record.get('purposes', ['all']),
                'categories': record.get('categories', ['all']),
                'timestamp': record.get('timestamp'),
            }

    def is_opted_out(self,
                     citizen_id: str,
                     purpose: str = 'research',
                     categories: Optional[List[str]] = None) -> bool:
        """
        Check if a citizen has opted out for given purpose/categories.

        Args:
            citizen_id: Pseudonymized citizen identifier
            purpose: The intended use purpose (research, ai_training, etc.)
            categories: Data categories being accessed

        Returns:
            True if citizen has opted out
        """
        if citizen_id not in self.opted_out:
            return False

        opt_out_info = self.opted_out[citizen_id]

        # Check purpose-specific opt-out
        if 'all' in opt_out_info['purposes'] or purpose in opt_out_info['purposes']:
            return True

        # Check category-specific opt-out
        if categories:
            opt_out_categories = opt_out_info.get('categories', [])
            if 'all' in opt_out_categories:
                return True
            if any(cat in opt_out_categories for cat in categories):
                return True

        return False

    def register_opt_out(self,
                         citizen_id: str,
                         purposes: List[str] = None,
                         categories: List[str] = None):
        """Register a new opt-out."""
        self.opted_out[citizen_id] = {
            'purposes': purposes or ['all'],
            'categories': categories or ['all'],
            'timestamp': datetime.now().isoformat()
        }

    def save_registry(self, filepath: str):
        """Save registry to file."""
        data = {
            'opt_outs': [
                {'citizen_id': cid, **info}
                for cid, info in self.opted_out.items()
            ],
            'last_updated': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)


# =============================================================================
# FHIR DATA LOADERS
# =============================================================================

class FHIRDataLoader(ABC):
    """Abstract base class for FHIR data loaders."""

    def __init__(self,
                 opt_out_registry: Optional[OptOutRegistry] = None,
                 purpose: str = 'research'):
        self.opt_out_registry = opt_out_registry or OptOutRegistry()
        self.purpose = purpose
        self.audit_log: List[Dict] = []

    @abstractmethod
    def load_patients(self, **kwargs) -> List[PatientRecord]:
        """Load patient records."""
        pass

    def to_fl_dataset(self,
                      records: List[PatientRecord],
                      feature_spec: List[str],
                      label_name: str,
                      apply_opt_out: bool = True) -> FLDataset:
        """
        Convert patient records to FL dataset.

        Args:
            records: List of PatientRecord objects
            feature_spec: List of feature names to extract
            label_name: Name of the label field
            apply_opt_out: Whether to filter opted-out records

        Returns:
            FLDataset ready for FL training
        """
        filtered_records = []

        for record in records:
            # Apply opt-out filtering
            if apply_opt_out and self.opt_out_registry.is_opted_out(
                record.pseudonymized_id,
                purpose=self.purpose
            ):
                continue

            filtered_records.append(record)

        # Log filtering
        self._log_access(
            total_records=len(records),
            filtered_records=len(filtered_records),
            opted_out=len(records) - len(filtered_records)
        )

        # Extract features and labels
        X_list = []
        y_list = []
        ids = []

        for record in filtered_records:
            X_list.append(record.to_feature_vector(feature_spec))

            # Get label
            label_map = {
                'mortality_30day': record.outcome_30day_mortality,
                'readmission': record.outcome_readmission,
                'icu_admission': record.outcome_icu_admission,
            }
            label = label_map.get(label_name, False)
            y_list.append(1 if label else 0)
            ids.append(record.pseudonymized_id)

        X = np.array(X_list, dtype=np.float32)
        y = np.array(y_list, dtype=np.int32)

        return FLDataset(
            X=X,
            y=y,
            patient_ids=ids,
            feature_names=feature_spec,
            label_name=label_name,
            metadata={
                'source': self.__class__.__name__,
                'purpose': self.purpose,
                'opt_out_applied': apply_opt_out,
                'original_count': len(records),
                'filtered_count': len(filtered_records)
            }
        )

    def _log_access(self, **kwargs):
        """Log data access for GDPR Article 30 compliance."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'purpose': self.purpose,
            **kwargs
        }
        self.audit_log.append(log_entry)

    def _pseudonymize(self, patient_id: str, salt: str = 'FL-EHDS') -> str:
        """Create pseudonymized ID from patient identifier."""
        combined = f"{salt}:{patient_id}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


class FHIRBundleLoader(FHIRDataLoader):
    """Load FHIR data from Bundle JSON files."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def load_patients(self,
                      bundle_path: str,
                      **kwargs) -> List[PatientRecord]:
        """
        Load patients from a FHIR Bundle JSON file.

        Args:
            bundle_path: Path to FHIR Bundle JSON file

        Returns:
            List of PatientRecord objects
        """
        with open(bundle_path, 'r') as f:
            bundle_data = json.load(f)

        return self._parse_bundle(bundle_data)

    def _parse_bundle(self, bundle_data: Dict) -> List[PatientRecord]:
        """Parse FHIR Bundle and extract patient records."""
        records = []

        # Index resources by type and reference
        resources_by_type: Dict[str, List[Dict]] = {}
        for entry in bundle_data.get('entry', []):
            resource = entry.get('resource', {})
            resource_type = resource.get('resourceType')

            if resource_type not in resources_by_type:
                resources_by_type[resource_type] = []
            resources_by_type[resource_type].append(resource)

        # Process each patient
        for patient_resource in resources_by_type.get('Patient', []):
            patient_id = patient_resource.get('id')

            record = PatientRecord(
                patient_id=patient_id,
                pseudonymized_id=self._pseudonymize(patient_id)
            )

            # Extract demographics
            record.gender = patient_resource.get('gender')

            if 'birthDate' in patient_resource:
                birth_date = datetime.strptime(
                    patient_resource['birthDate'], '%Y-%m-%d'
                ).date()
                record.birth_date = birth_date
                record.age = (date.today() - birth_date).days / 365.25

            # Find related observations
            for obs in resources_by_type.get('Observation', []):
                subject_ref = obs.get('subject', {}).get('reference', '')
                if patient_id not in subject_ref:
                    continue

                # Extract observation values
                code = obs.get('code', {}).get('coding', [{}])[0].get('code', '')
                value = obs.get('valueQuantity', {}).get('value')

                if value is not None:
                    # Map LOINC codes to fields
                    if code == '39156-5':  # BMI
                        record.bmi = value
                    elif code == '8480-6':  # Systolic BP
                        record.systolic_bp = value
                    elif code == '8462-4':  # Diastolic BP
                        record.diastolic_bp = value
                    elif code == '8867-4':  # Heart rate
                        record.heart_rate = value
                    elif code == '2339-0':  # Glucose
                        record.glucose = value
                    elif code == '2093-3':  # Cholesterol
                        record.cholesterol = value

            # Derive mortality label from Patient resource
            if patient_resource.get('deceasedBoolean') is True:
                record.outcome_30day_mortality = True
            elif patient_resource.get('deceasedDateTime'):
                record.outcome_30day_mortality = True
            else:
                record.outcome_30day_mortality = False

            # Find related conditions
            for condition in resources_by_type.get('Condition', []):
                subject_ref = condition.get('subject', {}).get('reference', '')
                if patient_id not in subject_ref:
                    continue

                code = condition.get('code', {}).get('coding', [{}])[0].get('code', '')
                if code:
                    record.conditions.append(code)

            # Parse MedicationRequest / MedicationStatement
            for med_type in ('MedicationRequest', 'MedicationStatement'):
                for med in resources_by_type.get(med_type, []):
                    subject_ref = med.get('subject', {}).get('reference', '')
                    if patient_id not in subject_ref:
                        continue
                    code = med.get('medicationCodeableConcept', {}).get(
                        'coding', [{}]
                    )[0].get('code', '')
                    if code:
                        record.medications.append(code)

            # Parse Encounter resources for readmission/ICU detection
            patient_encounters = []
            for enc in resources_by_type.get('Encounter', []):
                subject_ref = enc.get('subject', {}).get('reference', '')
                if patient_id not in subject_ref:
                    continue
                enc_class = enc.get('class', {}).get('code', '')
                period = enc.get('period', {})
                patient_encounters.append({
                    'class': enc_class,
                    'start': period.get('start'),
                    'end': period.get('end'),
                })

            # Detect readmission: >=2 inpatient encounters
            inpatient_encounters = [
                e for e in patient_encounters if e['class'] in ('IMP', 'EMER')
            ]
            if len(inpatient_encounters) >= 2:
                record.outcome_readmission = True
            else:
                record.outcome_readmission = False

            records.append(record)

        return records


class FHIRServerLoader(FHIRDataLoader):
    """Load FHIR data from a FHIR R4 server."""

    def __init__(self,
                 server_url: str,
                 auth_token: Optional[str] = None,
                 **kwargs):
        super().__init__(**kwargs)

        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required for FHIR server connection")

        self.server_url = server_url.rstrip('/')
        self.session = requests.Session()

        if auth_token:
            self.session.headers['Authorization'] = f'Bearer {auth_token}'

        self.session.headers['Accept'] = 'application/fhir+json'

    def load_patients(self,
                      query_params: Optional[Dict] = None,
                      max_patients: int = 100,
                      **kwargs) -> List[PatientRecord]:
        """
        Query FHIR server for patients.

        Args:
            query_params: FHIR search parameters
            max_patients: Maximum number of patients to load

        Returns:
            List of PatientRecord objects
        """
        params = query_params or {}
        params['_count'] = min(max_patients, 100)

        url = f"{self.server_url}/Patient"
        response = self.session.get(url, params=params)
        response.raise_for_status()

        bundle = response.json()
        records = []

        for entry in bundle.get('entry', []):
            patient_resource = entry.get('resource', {})
            patient_id = patient_resource.get('id')

            # Create basic record
            record = PatientRecord(
                patient_id=patient_id,
                pseudonymized_id=self._pseudonymize(patient_id)
            )

            # Extract demographics
            record.gender = patient_resource.get('gender')

            if 'birthDate' in patient_resource:
                birth_date = datetime.strptime(
                    patient_resource['birthDate'], '%Y-%m-%d'
                ).date()
                record.birth_date = birth_date
                record.age = (date.today() - birth_date).days / 365.25

            # Fetch observations for this patient
            obs_url = f"{self.server_url}/Observation"
            obs_response = self.session.get(
                obs_url,
                params={'patient': patient_id, '_count': 100}
            )

            if obs_response.status_code == 200:
                obs_bundle = obs_response.json()
                for obs_entry in obs_bundle.get('entry', []):
                    obs = obs_entry.get('resource', {})
                    code = obs.get('code', {}).get('coding', [{}])[0].get('code', '')
                    value = obs.get('valueQuantity', {}).get('value')

                    if value is not None:
                        if code == '39156-5':
                            record.bmi = value
                        elif code == '8480-6':
                            record.systolic_bp = value
                        elif code == '8867-4':
                            record.heart_rate = value

            records.append(record)

            if len(records) >= max_patients:
                break

        return records


class SyntheticFHIRLoader(FHIRDataLoader):
    """Generate synthetic FHIR-like data for testing."""

    def __init__(self,
                 num_patients: int = 500,
                 random_seed: int = 42,
                 hospital_profile: str = 'general',
                 **kwargs):
        super().__init__(**kwargs)

        self.num_patients = num_patients
        self.rng = np.random.RandomState(random_seed)
        self.hospital_profile = hospital_profile

        # Hospital profiles affect data distributions
        self.profiles = {
            'general': {
                'age_mean': 55, 'age_std': 18,
                'bmi_mean': 26, 'bmi_std': 5,
                'mortality_rate': 0.08
            },
            'cardiac': {
                'age_mean': 65, 'age_std': 12,
                'bmi_mean': 28, 'bmi_std': 4,
                'mortality_rate': 0.12
            },
            'pediatric': {
                'age_mean': 8, 'age_std': 5,
                'bmi_mean': 18, 'bmi_std': 3,
                'mortality_rate': 0.02
            },
            'geriatric': {
                'age_mean': 78, 'age_std': 8,
                'bmi_mean': 24, 'bmi_std': 4,
                'mortality_rate': 0.15
            },
            'oncology': {
                'age_mean': 60, 'age_std': 15,
                'bmi_mean': 25, 'bmi_std': 5,
                'mortality_rate': 0.20
            }
        }

    def load_patients(self, **kwargs) -> List[PatientRecord]:
        """Generate synthetic patient records."""
        profile = self.profiles.get(self.hospital_profile, self.profiles['general'])
        records = []

        for i in range(self.num_patients):
            patient_id = f"synthetic-{i:06d}"

            # Demographics
            age = max(0, self.rng.normal(profile['age_mean'], profile['age_std']))
            gender = self.rng.choice(['male', 'female'])

            # Clinical measurements
            bmi = max(15, self.rng.normal(profile['bmi_mean'], profile['bmi_std']))
            systolic = max(80, self.rng.normal(120 + age * 0.3, 15))
            diastolic = max(50, self.rng.normal(80, 10))
            heart_rate = max(40, self.rng.normal(75, 12))
            glucose = max(60, self.rng.normal(100 + age * 0.2, 25))
            cholesterol = max(100, self.rng.normal(200 + age * 0.5, 40))

            # Outcome (influenced by risk factors)
            risk_score = (
                (age - 50) * 0.02 +
                (bmi - 25) * 0.01 +
                (systolic - 120) * 0.005 +
                (glucose - 100) * 0.002
            )
            mortality_prob = min(0.9, max(0.01, profile['mortality_rate'] + risk_score))
            mortality = self.rng.random() < mortality_prob

            record = PatientRecord(
                patient_id=patient_id,
                pseudonymized_id=self._pseudonymize(patient_id),
                age=age,
                gender=gender,
                bmi=bmi,
                systolic_bp=systolic,
                diastolic_bp=diastolic,
                heart_rate=heart_rate,
                glucose=glucose,
                cholesterol=cholesterol,
                outcome_30day_mortality=mortality,
                source_hospital=self.hospital_profile
            )

            records.append(record)

        return records


# =============================================================================
# FL NODE DATA MANAGER
# =============================================================================

class FLNodeDataManager:
    """
    Manages data loading for multiple FL nodes with different FHIR sources.

    This is the main entry point for FL training data preparation.
    """

    def __init__(self,
                 opt_out_registry_path: Optional[str] = None,
                 purpose: str = 'research'):
        self.opt_out_registry = OptOutRegistry(opt_out_registry_path)
        self.purpose = purpose
        self.node_loaders: Dict[int, FHIRDataLoader] = {}

    def add_node_synthetic(self,
                           node_id: int,
                           num_patients: int = 500,
                           hospital_profile: str = 'general',
                           random_seed: Optional[int] = None):
        """Add a node with synthetic data."""
        seed = random_seed or (42 + node_id * 100)

        self.node_loaders[node_id] = SyntheticFHIRLoader(
            num_patients=num_patients,
            random_seed=seed,
            hospital_profile=hospital_profile,
            opt_out_registry=self.opt_out_registry,
            purpose=self.purpose
        )

    def add_node_fhir_server(self,
                             node_id: int,
                             server_url: str,
                             auth_token: Optional[str] = None):
        """Add a node connected to a FHIR server."""
        self.node_loaders[node_id] = FHIRServerLoader(
            server_url=server_url,
            auth_token=auth_token,
            opt_out_registry=self.opt_out_registry,
            purpose=self.purpose
        )

    def add_node_bundle(self,
                        node_id: int,
                        bundle_path: str):
        """Add a node with FHIR bundle file."""
        self.node_loaders[node_id] = FHIRBundleLoader(
            opt_out_registry=self.opt_out_registry,
            purpose=self.purpose
        )
        # Store path for later loading
        self.node_loaders[node_id]._bundle_path = bundle_path

    def load_all_nodes(self,
                       feature_spec: List[str],
                       label_name: str = 'mortality_30day',
                       normalize: bool = True) -> Dict[int, FLDataset]:
        """
        Load datasets for all configured nodes.

        Args:
            feature_spec: List of features to extract
            label_name: Target label name
            normalize: Whether to normalize features

        Returns:
            Dictionary mapping node_id to FLDataset
        """
        datasets = {}

        for node_id, loader in self.node_loaders.items():
            # Load patient records
            if isinstance(loader, SyntheticFHIRLoader):
                records = loader.load_patients()
            elif hasattr(loader, '_bundle_path'):
                records = loader.load_patients(bundle_path=loader._bundle_path)
            else:
                records = loader.load_patients()

            # Convert to FL dataset
            dataset = loader.to_fl_dataset(
                records,
                feature_spec=feature_spec,
                label_name=label_name,
                apply_opt_out=True
            )

            if normalize:
                dataset = dataset.normalize(method='standard')

            datasets[node_id] = dataset

        return datasets


# =============================================================================
# FHIR DATA BRIDGE FOR FL TRAINING
# =============================================================================

DEFAULT_HOSPITAL_PROFILES = ['general', 'cardiac', 'pediatric', 'geriatric', 'oncology']

DEFAULT_FEATURE_SPEC = [
    'age', 'gender', 'bmi', 'systolic_bp', 'diastolic_bp',
    'heart_rate', 'glucose', 'cholesterol', 'num_conditions', 'num_medications'
]


def load_fhir_data(
    num_clients: int = 5,
    samples_per_client: int = 500,
    hospital_profiles: Optional[List[str]] = None,
    bundle_paths: Optional[Dict[int, str]] = None,
    feature_spec: Optional[List[str]] = None,
    label_name: str = 'mortality_30day',
    opt_out_registry_path: Optional[str] = None,
    purpose: str = 'ai_training',
    test_split: float = 0.2,
    seed: int = 42,
) -> Tuple[Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[int, Tuple[np.ndarray, np.ndarray]],
           Dict[str, Any]]:
    """
    Load FHIR data for federated learning, returning the same format
    as generate_healthcare_data().

    Hospital profiles create natural non-IID distributions:
    - general: age~55, mortality~8%
    - cardiac: age~65, mortality~12%
    - pediatric: age~8, mortality~2%
    - geriatric: age~78, mortality~15%
    - oncology: age~60, mortality~20%

    Args:
        num_clients: Number of FL clients (hospitals)
        samples_per_client: Patients per hospital
        hospital_profiles: Profile per client (cycles if fewer than num_clients)
        bundle_paths: Optional dict mapping client_id -> FHIR Bundle JSON path
        feature_spec: Features to extract from FHIR records
        label_name: Target label (mortality_30day, readmission, icu_admission)
        opt_out_registry_path: Path to EHDS Article 71 opt-out registry
        purpose: EHDS purpose for data access
        test_split: Fraction of data for testing (per client)
        seed: Random seed

    Returns:
        (client_train_data, client_test_data, metadata)
        where each data dict maps client_id -> (X: np.ndarray, y: np.ndarray)
    """
    profiles = hospital_profiles or DEFAULT_HOSPITAL_PROFILES
    features = feature_spec or DEFAULT_FEATURE_SPEC
    bundle_paths = bundle_paths or {}

    # Assign profiles to clients (cycle if fewer profiles than clients)
    assigned_profiles = [profiles[i % len(profiles)] for i in range(num_clients)]

    # Create node manager
    manager = FLNodeDataManager(
        opt_out_registry_path=opt_out_registry_path,
        purpose=purpose
    )

    # Configure nodes
    for node_id in range(num_clients):
        if node_id in bundle_paths:
            manager.add_node_bundle(node_id, bundle_paths[node_id])
        else:
            manager.add_node_synthetic(
                node_id=node_id,
                num_patients=samples_per_client,
                hospital_profile=assigned_profiles[node_id],
                random_seed=seed + node_id * 100
            )

    # Load all node datasets
    datasets = manager.load_all_nodes(
        feature_spec=features,
        label_name=label_name,
        normalize=True
    )

    # Split each node's data into train/test
    rng = np.random.RandomState(seed)
    client_train_data = {}
    client_test_data = {}
    total_opted_out = 0

    for node_id, dataset in datasets.items():
        n = dataset.n_samples
        n_test = max(1, int(n * test_split))
        perm = rng.permutation(n)
        train_idx = perm[:-n_test]
        test_idx = perm[-n_test:]

        client_train_data[node_id] = (
            dataset.X[train_idx].astype(np.float32),
            dataset.y[train_idx].astype(np.int64)
        )
        client_test_data[node_id] = (
            dataset.X[test_idx].astype(np.float32),
            dataset.y[test_idx].astype(np.int64)
        )

        # Track opt-out stats from metadata
        opted_out = dataset.metadata.get('original_count', n) - dataset.metadata.get('filtered_count', n)
        total_opted_out += opted_out

    metadata = {
        'feature_names': features,
        'num_features': len(features),
        'label_name': label_name,
        'profiles_assigned': assigned_profiles,
        'samples_per_client': samples_per_client,
        'total_opted_out': total_opted_out,
        'source': 'fhir_loader',
        'test_split': test_split,
    }

    return client_train_data, client_test_data, metadata


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

def example_usage():
    """Demonstrate how to use the FHIR loader for FL training."""
    print("FL-EHDS FHIR Data Loader Example")
    print("=" * 50)

    # Create node data manager
    manager = FLNodeDataManager(purpose='ai_training')

    # Configure nodes with different hospital profiles
    # This creates realistic Non-IID data distributions
    hospital_profiles = [
        ('general', 400),
        ('cardiac', 350),
        ('pediatric', 300),
        ('geriatric', 450),
        ('oncology', 380)
    ]

    for node_id, (profile, n_patients) in enumerate(hospital_profiles):
        manager.add_node_synthetic(
            node_id=node_id,
            num_patients=n_patients,
            hospital_profile=profile,
            random_seed=42 + node_id
        )

    # Define features to extract
    feature_spec = [
        'age', 'gender', 'bmi', 'systolic_bp',
        'heart_rate', 'glucose', 'cholesterol'
    ]

    # Load all node datasets
    datasets = manager.load_all_nodes(
        feature_spec=feature_spec,
        label_name='mortality_30day',
        normalize=True
    )

    # Print statistics
    print("\nNode Data Statistics:")
    print("-" * 50)

    for node_id, dataset in datasets.items():
        print(f"\nNode {node_id + 1}:")
        print(f"  Samples: {dataset.n_samples}")
        print(f"  Features: {dataset.n_features}")
        print(f"  Positive rate: {dataset.y.mean():.2%}")
        print(f"  Feature means: {dataset.X.mean(axis=0)[:3].round(2)}")

    return datasets


if __name__ == "__main__":
    datasets = example_usage()
