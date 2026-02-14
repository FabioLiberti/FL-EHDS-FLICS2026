#!/usr/bin/env python3
"""
Generate Synthea-style FHIR R4 Bundle JSON files for FL-EHDS testing.

Creates 5 hospital bundles with realistic clinical data distributions:
- hospital_general.json   — age~55, mixed conditions
- hospital_cardiac.json   — age~65, cardiac-heavy
- hospital_pediatric.json — age~8, pediatric conditions
- hospital_geriatric.json — age~78, multi-morbidity
- hospital_oncology.json  — age~60, cancer diagnoses

Each bundle contains ~100 patients with linked Observations, Conditions,
MedicationRequests, and Encounters — all valid FHIR R4 JSON.

Usage:
    python data/fhir_bundles/generate_bundles.py

Author: Fabio Liberti
"""

import json
import uuid
import numpy as np
from pathlib import Path
from datetime import date, timedelta


# =========================================================================
# CLINICAL CODE TABLES
# =========================================================================

LOINC_OBSERVATIONS = {
    "39156-5": {"display": "Body mass index", "unit": "kg/m2"},
    "8480-6": {"display": "Systolic blood pressure", "unit": "mmHg"},
    "8462-4": {"display": "Diastolic blood pressure", "unit": "mmHg"},
    "8867-4": {"display": "Heart rate", "unit": "/min"},
    "2339-0": {"display": "Glucose [Mass/volume] in Blood", "unit": "mg/dL"},
    "2093-3": {"display": "Cholesterol [Mass/volume] in Serum", "unit": "mg/dL"},
    "718-7": {"display": "Hemoglobin [Mass/volume] in Blood", "unit": "g/dL"},
    "2160-0": {"display": "Creatinine [Mass/volume] in Serum", "unit": "mg/dL"},
}

# Conditions pool: (SNOMED code, display, ICD-10 code, category)
CONDITIONS_GENERAL = [
    ("73211009", "Diabetes mellitus", "E11", "metabolic"),
    ("38341003", "Hypertensive disorder", "I10", "cardiac"),
    ("13645005", "COPD", "J44", "respiratory"),
    ("414545008", "Ischemic heart disease", "I25", "cardiac"),
    ("235856003", "Hepatopathy", "K76", "hepatic"),
    ("40930008", "Hypothyroidism", "E03", "endocrine"),
    ("195967001", "Asthma", "J45", "respiratory"),
    ("49436004", "Atrial fibrillation", "I48", "cardiac"),
]

CONDITIONS_CARDIAC = [
    ("414545008", "Ischemic heart disease", "I25", "cardiac"),
    ("84114007", "Heart failure", "I50", "cardiac"),
    ("49436004", "Atrial fibrillation", "I48", "cardiac"),
    ("22298006", "Myocardial infarction", "I21", "cardiac"),
    ("38341003", "Hypertensive disorder", "I10", "cardiac"),
    ("73211009", "Diabetes mellitus", "E11", "metabolic"),
    ("44054006", "Diabetes mellitus type 2", "E11.9", "metabolic"),
    ("698247007", "Cardiac arrhythmia", "I49", "cardiac"),
]

CONDITIONS_PEDIATRIC = [
    ("195967001", "Asthma", "J45", "respiratory"),
    ("386661006", "Fever", "R50", "general"),
    ("65363002", "Otitis media", "H66", "ent"),
    ("36971009", "Sinusitis", "J32", "respiratory"),
    ("25064002", "Headache", "R51", "neurological"),
    ("87433001", "Allergic rhinitis", "J30", "respiratory"),
    ("431855005", "Obesity in childhood", "E66", "metabolic"),
    ("128613002", "Seizure disorder", "G40", "neurological"),
]

CONDITIONS_GERIATRIC = [
    ("73211009", "Diabetes mellitus", "E11", "metabolic"),
    ("38341003", "Hypertensive disorder", "I10", "cardiac"),
    ("84114007", "Heart failure", "I50", "cardiac"),
    ("396275006", "Osteoarthritis", "M15", "musculoskeletal"),
    ("26929004", "Alzheimer disease", "G30", "neurological"),
    ("13645005", "COPD", "J44", "respiratory"),
    ("431855005", "Obesity", "E66", "metabolic"),
    ("40930008", "Hypothyroidism", "E03", "endocrine"),
]

CONDITIONS_ONCOLOGY = [
    ("254637007", "Lung cancer", "C34", "oncology"),
    ("254838004", "Breast cancer", "C50", "oncology"),
    ("363406005", "Colon cancer", "C18", "oncology"),
    ("399068003", "Prostate cancer", "C61", "oncology"),
    ("93761005", "Primary liver cancer", "C22", "oncology"),
    ("73211009", "Diabetes mellitus", "E11", "metabolic"),
    ("38341003", "Hypertensive disorder", "I10", "cardiac"),
    ("271737000", "Anemia", "D64", "hematological"),
]

# Medications: (ATC code, display)
MEDICATIONS_GENERAL = [
    ("C09AA02", "Enalapril"), ("C10AA01", "Simvastatin"),
    ("A10BA02", "Metformin"), ("N02BE01", "Paracetamol"),
    ("A02BC01", "Omeprazole"), ("B01AC06", "Aspirin"),
]

MEDICATIONS_CARDIAC = [
    ("C09AA02", "Enalapril"), ("C10AA01", "Simvastatin"),
    ("B01AC06", "Aspirin"), ("C07AB02", "Metoprolol"),
    ("C01AA05", "Digoxin"), ("C03CA01", "Furosemide"),
    ("B01AA03", "Warfarin"), ("C08CA01", "Amlodipine"),
]

MEDICATIONS_PEDIATRIC = [
    ("N02BE01", "Paracetamol"), ("J01CA04", "Amoxicillin"),
    ("R03AC02", "Salbutamol"), ("R06AE07", "Cetirizine"),
]

MEDICATIONS_GERIATRIC = [
    ("C09AA02", "Enalapril"), ("C10AA01", "Simvastatin"),
    ("A10BA02", "Metformin"), ("M01AE01", "Ibuprofen"),
    ("N06DA02", "Donepezil"), ("A02BC01", "Omeprazole"),
    ("C03CA01", "Furosemide"), ("B01AC06", "Aspirin"),
]

MEDICATIONS_ONCOLOGY = [
    ("L01XA01", "Cisplatin"), ("L01BC02", "Fluorouracil"),
    ("L02BG04", "Letrozole"), ("N02AA01", "Morphine"),
    ("A04AA01", "Ondansetron"), ("H02AB06", "Prednisolone"),
]


# =========================================================================
# HOSPITAL PROFILE DEFINITIONS
# =========================================================================

PROFILES = {
    "general": {
        "age_mean": 55, "age_std": 18,
        "bmi_mean": 26, "bmi_std": 5,
        "mortality_rate": 0.08,
        "conditions": CONDITIONS_GENERAL,
        "medications": MEDICATIONS_GENERAL,
        "n_conditions_range": (0, 4),
        "n_medications_range": (0, 4),
        "country": "IT",
    },
    "cardiac": {
        "age_mean": 65, "age_std": 12,
        "bmi_mean": 28, "bmi_std": 4,
        "mortality_rate": 0.12,
        "conditions": CONDITIONS_CARDIAC,
        "medications": MEDICATIONS_CARDIAC,
        "n_conditions_range": (1, 5),
        "n_medications_range": (1, 5),
        "country": "DE",
    },
    "pediatric": {
        "age_mean": 8, "age_std": 5,
        "bmi_mean": 18, "bmi_std": 3,
        "mortality_rate": 0.02,
        "conditions": CONDITIONS_PEDIATRIC,
        "medications": MEDICATIONS_PEDIATRIC,
        "n_conditions_range": (0, 2),
        "n_medications_range": (0, 2),
        "country": "FR",
    },
    "geriatric": {
        "age_mean": 78, "age_std": 8,
        "bmi_mean": 24, "bmi_std": 4,
        "mortality_rate": 0.15,
        "conditions": CONDITIONS_GERIATRIC,
        "medications": MEDICATIONS_GERIATRIC,
        "n_conditions_range": (2, 6),
        "n_medications_range": (2, 6),
        "country": "IT",
    },
    "oncology": {
        "age_mean": 60, "age_std": 15,
        "bmi_mean": 25, "bmi_std": 5,
        "mortality_rate": 0.20,
        "conditions": CONDITIONS_ONCOLOGY,
        "medications": MEDICATIONS_ONCOLOGY,
        "n_conditions_range": (1, 4),
        "n_medications_range": (1, 4),
        "country": "DE",
    },
}


# =========================================================================
# BUNDLE GENERATION
# =========================================================================

def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def generate_patient_bundle(
    profile_name: str,
    n_patients: int = 100,
    seed: int = 42,
) -> dict:
    """Generate a FHIR R4 Bundle for a hospital profile."""
    rng = np.random.RandomState(seed)
    profile = PROFILES[profile_name]
    entries = []
    obs_counter = 0
    cond_counter = 0
    med_counter = 0
    enc_counter = 0

    for i in range(n_patients):
        patient_id = f"{profile_name[:3]}-p-{i:04d}"

        # --- Demographics ---
        age = max(0.5, rng.normal(profile["age_mean"], profile["age_std"]))
        gender = rng.choice(["male", "female"])
        birth_year = date.today().year - int(age)
        birth_month = rng.randint(1, 13)
        birth_day = rng.randint(1, 29)
        birth_date = f"{birth_year:04d}-{birth_month:02d}-{birth_day:02d}"

        # --- Clinical values ---
        bmi = max(12.0, rng.normal(profile["bmi_mean"], profile["bmi_std"]))
        systolic = max(70.0, rng.normal(120 + age * 0.3, 15))
        diastolic = max(40.0, rng.normal(80, 10))
        heart_rate = max(40.0, rng.normal(75, 12))
        glucose = max(50.0, rng.normal(100 + age * 0.2, 25))
        cholesterol = max(80.0, rng.normal(200 + age * 0.5, 40))

        # --- Conditions ---
        n_conds = rng.randint(profile["n_conditions_range"][0],
                              profile["n_conditions_range"][1] + 1)
        patient_conditions = []
        if n_conds > 0:
            cond_indices = rng.choice(
                len(profile["conditions"]),
                size=min(n_conds, len(profile["conditions"])),
                replace=False,
            )
            patient_conditions = [profile["conditions"][j] for j in cond_indices]

        # --- Mortality (logistic risk model) ---
        risk_score = (
            (age - 50) * 0.02
            + (bmi - 25) * 0.01
            + (systolic - 120) * 0.005
            + (glucose - 100) * 0.002
            + len(patient_conditions) * 0.03
        )
        mortality_prob = float(_sigmoid(
            np.log(profile["mortality_rate"] / (1 - profile["mortality_rate"]))
            + risk_score
        ))
        is_deceased = bool(rng.random() < mortality_prob)

        # --- Medications ---
        n_meds = rng.randint(profile["n_medications_range"][0],
                             profile["n_medications_range"][1] + 1)
        patient_meds = []
        if n_meds > 0:
            med_indices = rng.choice(
                len(profile["medications"]),
                size=min(n_meds, len(profile["medications"])),
                replace=False,
            )
            patient_meds = [profile["medications"][j] for j in med_indices]

        # ====== PATIENT RESOURCE ======
        patient_resource = {
            "resourceType": "Patient",
            "id": patient_id,
            "gender": gender,
            "birthDate": birth_date,
            "deceasedBoolean": is_deceased,
            "address": [{"country": profile["country"]}],
        }
        entries.append({"resource": patient_resource})

        # ====== OBSERVATION RESOURCES ======
        obs_values = {
            "39156-5": round(bmi, 1),
            "8480-6": round(systolic, 0),
            "8462-4": round(diastolic, 0),
            "8867-4": round(heart_rate, 0),
            "2339-0": round(glucose, 1),
            "2093-3": round(cholesterol, 1),
        }

        for loinc_code, value in obs_values.items():
            info = LOINC_OBSERVATIONS[loinc_code]
            obs_id = f"{profile_name[:3]}-obs-{obs_counter:06d}"
            obs_counter += 1

            # Random observation date in the last year
            days_ago = rng.randint(1, 365)
            obs_date = (date.today() - timedelta(days=int(days_ago))).isoformat()

            obs_resource = {
                "resourceType": "Observation",
                "id": obs_id,
                "status": "final",
                "subject": {"reference": f"Patient/{patient_id}"},
                "effectiveDateTime": obs_date,
                "code": {
                    "coding": [{
                        "system": "http://loinc.org",
                        "code": loinc_code,
                        "display": info["display"],
                    }]
                },
                "valueQuantity": {
                    "value": value,
                    "unit": info["unit"],
                    "system": "http://unitsofmeasure.org",
                },
            }
            entries.append({"resource": obs_resource})

        # ====== CONDITION RESOURCES ======
        for snomed, display, icd10, category in patient_conditions:
            cond_id = f"{profile_name[:3]}-cond-{cond_counter:06d}"
            cond_counter += 1

            cond_resource = {
                "resourceType": "Condition",
                "id": cond_id,
                "subject": {"reference": f"Patient/{patient_id}"},
                "clinicalStatus": {
                    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical",
                                "code": "active"}]
                },
                "code": {
                    "coding": [
                        {"system": "http://snomed.info/sct", "code": snomed, "display": display},
                        {"system": "http://hl7.org/fhir/sid/icd-10", "code": icd10},
                    ]
                },
                "category": [{
                    "coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-category",
                                "code": "encounter-diagnosis"}]
                }],
            }
            entries.append({"resource": cond_resource})

        # ====== MEDICATION RESOURCES ======
        for atc_code, med_display in patient_meds:
            med_id = f"{profile_name[:3]}-med-{med_counter:06d}"
            med_counter += 1

            med_resource = {
                "resourceType": "MedicationRequest",
                "id": med_id,
                "status": "active",
                "intent": "order",
                "subject": {"reference": f"Patient/{patient_id}"},
                "medicationCodeableConcept": {
                    "coding": [{
                        "system": "http://www.whocc.no/atc",
                        "code": atc_code,
                        "display": med_display,
                    }]
                },
            }
            entries.append({"resource": med_resource})

        # ====== ENCOUNTER RESOURCES ======
        n_encounters = rng.choice([0, 1, 1, 2], p=[0.2, 0.4, 0.3, 0.1])
        for _ in range(n_encounters):
            enc_id = f"{profile_name[:3]}-enc-{enc_counter:06d}"
            enc_counter += 1

            enc_class = rng.choice(["IMP", "AMB", "EMER"], p=[0.4, 0.4, 0.2])
            days_ago = rng.randint(1, 365)
            start_date = date.today() - timedelta(days=int(days_ago))
            los = int(rng.choice([1, 2, 3, 5, 7, 14], p=[0.3, 0.25, 0.2, 0.1, 0.1, 0.05]))
            end_date = start_date + timedelta(days=los)

            enc_resource = {
                "resourceType": "Encounter",
                "id": enc_id,
                "status": "finished",
                "class": {
                    "system": "http://terminology.hl7.org/CodeSystem/v3-ActCode",
                    "code": enc_class,
                },
                "subject": {"reference": f"Patient/{patient_id}"},
                "period": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat(),
                },
            }
            entries.append({"resource": enc_resource})

    # Assemble bundle
    bundle = {
        "resourceType": "Bundle",
        "id": str(uuid.uuid4()),
        "type": "collection",
        "timestamp": date.today().isoformat() + "T00:00:00Z",
        "total": n_patients,
        "entry": entries,
    }

    return bundle


def main():
    """Generate all 5 hospital bundles."""
    out_dir = Path(__file__).parent

    for i, (profile_name, profile) in enumerate(PROFILES.items()):
        filename = f"hospital_{profile_name}.json"
        filepath = out_dir / filename

        bundle = generate_patient_bundle(
            profile_name=profile_name,
            n_patients=100,
            seed=42 + i * 100,
        )

        with open(filepath, "w") as f:
            json.dump(bundle, f, indent=2)

        # Count resources
        resource_counts = {}
        for entry in bundle["entry"]:
            rt = entry["resource"]["resourceType"]
            resource_counts[rt] = resource_counts.get(rt, 0) + 1

        n_deceased = sum(
            1 for e in bundle["entry"]
            if e["resource"].get("resourceType") == "Patient"
            and e["resource"].get("deceasedBoolean") is True
        )

        print(f"  {filename}: {resource_counts}, "
              f"mortality={n_deceased}/{bundle['total']} "
              f"({n_deceased/bundle['total']:.0%})")

    print(f"\nGenerated {len(PROFILES)} bundles in {out_dir}/")


if __name__ == "__main__":
    main()
