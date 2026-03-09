#!/usr/bin/env python3
"""
Reviewer Objection Mitigation — Cascade Script 2
==================================================
5 phases ordered by increasing execution time.  Each phase addresses a
specific reviewer concern.

Phase 2A : Governance overhead benchmark          (~1-2 h,  MEDIO)
           Measures per-round overhead of EHDS governance hooks
           (permit validation, audit logging, data minimization).

Phase 1B : Data quality stress testing            (~4-6 h,  MEDIO)
           Robustness to label noise, feature noise, missing features.

Phase 5B : HPFL split-point ablation              (~6-8 h,  ALTO)
           Ablates HPFL personal/shared boundary on imaging.
           depth=1 classifier only, depth=2 +layer4, depth=3 +layer3+layer4.

Phase 4A : EfficientNet-B0 imaging                (~8-12 h, ALTO)
           Architecture generality: EfficientNet-B0 on 3 imaging datasets.

Phase 3A : TabNet federated (tabular)             (~12-20 h, MOLTO ALTO)
           Architecture generality: attention-based TabNet on 3+ tabular datasets.

Usage:
  Quick validation (~27 exp.):
    cd fl-ehds-framework && python -m benchmarks.run_reviewer_cascade_2 --quick

  Full run (~225+ exp.):
    cd fl-ehds-framework && python -m benchmarks.run_reviewer_cascade_2 --full

  Resume from phase:
    cd fl-ehds-framework && python -m benchmarks.run_reviewer_cascade_2 --full --phase 5B
"""

import gc
import json
import math
import os
import shutil
import signal
import sys
import tempfile
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

FRAMEWORK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch
import torch.nn as nn

# Lazy imports for trainers — only import when needed to avoid heavy init
# from terminal.fl_trainer import FederatedTrainer
# from terminal.training.federated_image import ImageFederatedTrainer

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks" / "paper_results_cascade2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_FILE = "checkpoint_reviewer_cascade_2.json"

# ============================================================================
# LOGGING
# ============================================================================

def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)

# ============================================================================
# CHECKPOINT (atomic save with backup)
# ============================================================================

_checkpoint_data: Dict[str, Any] = {}

def save_checkpoint():
    _checkpoint_data["metadata"]["last_save"] = datetime.now().isoformat()
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    fd, tmp = tempfile.mkstemp(dir=str(OUTPUT_DIR), prefix=".rc2_", suffix=".tmp")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(_checkpoint_data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copy2(str(path), str(bak))
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def load_checkpoint():
    for p in [OUTPUT_DIR / CHECKPOINT_FILE, OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")]:
        if p.exists():
            try:
                with open(p) as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                continue
    return None

# ============================================================================
# GRACEFUL SHUTDOWN
# ============================================================================

_shutdown = False

def _signal_handler(signum, frame):
    global _shutdown
    if _shutdown:
        sys.exit(1)
    _shutdown = True
    log("INTERRUPT — saving checkpoint...")
    try:
        save_checkpoint()
    except Exception:
        pass
    sys.exit(0)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

# ============================================================================
# GPU CLEANUP
# ============================================================================

def _cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    gc.collect()

# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

SEEDS_3 = [42, 123, 456]
SEEDS_5 = [42, 123, 456, 789, 2024]

# --- Tabular datasets ---
TABULAR_DATASETS = {
    "Cardiovascular": {
        "loader": "cardiovascular",
        "input_dim": 11,
        "num_classes": 2,
        "num_clients": 5,
    },
    "Heart_Disease": {
        "loader": "heart_disease",
        "input_dim": 13,
        "num_classes": 2,
        "num_clients": 5,
    },
    "PTB-XL": {
        "loader": "ptbxl",
        "input_dim": 9,
        "num_classes": 5,
        "num_clients": 5,
    },
}

# --- Imaging datasets ---
IMAGING_DATASETS = {
    "Brain_Tumor": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Brain_Tumor"),
        "num_clients": 5,
        "num_classes": 4,
    },
    "Skin_Cancer": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Skin Cancer"),
        "num_clients": 5,
        "num_classes": 2,
    },
    "chest_xray": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "chest_xray"),
        "num_clients": 5,
        "num_classes": 2,
    },
}

IMAGING_LR_OVERRIDES = {
    "Brain_Tumor": {"learning_rate": 0.0005},
    "Skin_Cancer": {"learning_rate": 0.0005},
    "chest_xray": {"learning_rate": 0.0005},
}

ALGOS_CORE = ["FedAvg", "Ditto", "HPFL"]


# ============================================================================
# MODEL: HealthcareEfficientNet (Phase 4A)
# ============================================================================

class HealthcareEfficientNet(nn.Module):
    """
    EfficientNet-B0 for medical image classification in FL settings.

    Uses pretrained ImageNet weights with GroupNorm (FL-stable, replaces BatchNorm).
    Input: (batch, 3, 224, 224) - RGB images
    ~5.3M params total.
    """

    def __init__(
        self,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        import torchvision.models as tv_models

        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        base = tv_models.efficientnet_b0(weights=weights)

        # Replace all BatchNorm2d with GroupNorm (FL-stable)
        self._replace_bn_with_gn(base)

        # Backbone = features + avgpool
        self.features = base.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        # Classifier head: EfficientNet-B0 outputs 1280 features
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(1280, num_classes),
        )

    @staticmethod
    def _replace_bn_with_gn(module: nn.Module):
        """Recursively replace BatchNorm2d with GroupNorm."""
        for name, child in module.named_children():
            if isinstance(child, nn.BatchNorm2d):
                num_channels = child.num_features
                num_groups = min(16, num_channels)
                while num_channels % num_groups != 0 and num_groups > 1:
                    num_groups -= 1
                setattr(module, name, nn.GroupNorm(num_groups, num_channels))
            else:
                HealthcareEfficientNet._replace_bn_with_gn(child)

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# ============================================================================
# MODEL: HealthcareTabNet (Phase 3A)
# ============================================================================

class HealthcareTabNet(nn.Module):
    """
    Attention-based tabular network inspired by TabNet (Arik & Pfister, AAAI 2021).

    Pure PyTorch implementation (no pytorch-tabnet dependency).
    Uses sequential attention with prior scales for feature selection.

    Args:
        input_dim: number of input features
        num_classes: number of output classes
        n_d: dimension of decision step outputs
        n_a: dimension of attention step outputs
        n_steps: number of sequential attention steps
        gamma: coefficient for feature reuse in attention
        relaxation_factor: controls feature reuse penalty
    """

    def __init__(
        self,
        input_dim: int = 11,
        num_classes: int = 2,
        n_d: int = 64,
        n_a: int = 64,
        n_steps: int = 3,
        gamma: float = 1.3,
        relaxation_factor: float = 1.5,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.relaxation_factor = relaxation_factor

        # Initial batch normalization (replaced with LayerNorm for FL stability)
        self.initial_ln = nn.LayerNorm(input_dim)

        # Shared and step-specific layers
        self.shared_fc = nn.Linear(input_dim, n_d + n_a, bias=False)

        self.step_attn = nn.ModuleList()
        self.step_fc = nn.ModuleList()
        self.step_ln = nn.ModuleList()

        for _ in range(n_steps):
            self.step_attn.append(nn.Sequential(
                nn.Linear(n_a, input_dim, bias=False),
                # Sparsemax approximated via softmax with temperature
            ))
            self.step_fc.append(nn.Linear(input_dim, n_d + n_a, bias=False))
            self.step_ln.append(nn.LayerNorm(n_d + n_a))

        # Final classifier
        self.head = nn.Sequential(
            nn.Linear(n_d, num_classes),
        )

    def forward(self, x):
        # x: (batch, input_dim)
        x = self.initial_ln(x)
        batch_size = x.size(0)

        # Prior scales: tracks how much each feature has been used
        prior_scales = torch.ones(batch_size, self.input_dim, device=x.device)

        # Aggregated decision output
        agg_decision = torch.zeros(batch_size, self.n_d, device=x.device)

        # Initial shared transform
        h = self.shared_fc(x)  # (batch, n_d + n_a)

        for step in range(self.n_steps):
            # Split into decision and attention parts
            h_d = h[:, :self.n_d]       # decision
            h_a = h[:, self.n_d:]       # attention

            # Attention: compute feature importance mask
            attn_logits = self.step_attn[step](h_a)  # (batch, input_dim)
            attn_logits = attn_logits * prior_scales
            attn_mask = torch.softmax(attn_logits, dim=-1)  # (batch, input_dim)

            # Update prior scales (reduce importance of used features)
            prior_scales = prior_scales * (self.gamma - attn_mask)

            # Apply attention mask to input
            masked_x = attn_mask * x  # (batch, input_dim)

            # Step-specific transform
            h = self.step_fc[step](masked_x)
            h = self.step_ln[step](h)
            h = h * torch.sigmoid(h)  # GLU-like activation

            # Accumulate decision
            agg_decision = agg_decision + nn.functional.relu(h[:, :self.n_d])

        # Final classification
        out = self.head(agg_decision)
        return out


# ============================================================================
# DATA LOADING UTILITIES
# ============================================================================

_data_cache: Dict[str, Any] = {}

def _load_tabular(ds_name: str, seed: int):
    """Load tabular dataset. Returns (train, test, metadata)."""
    cfg = TABULAR_DATASETS[ds_name]
    cache_key = f"{ds_name}_s{seed}"
    if cache_key in _data_cache:
        c = _data_cache[cache_key]
        return (
            {cid: (X.copy(), y.copy()) for cid, (X, y) in c[0].items()},
            c[1],
            c[2],
        )

    loader_name = cfg["loader"]
    if loader_name == "cardiovascular":
        from data.cardiovascular_loader import load_cardiovascular_data
        train, test, meta = load_cardiovascular_data(
            num_clients=cfg["num_clients"], seed=seed
        )
    elif loader_name == "heart_disease":
        from data.heart_disease_loader import load_heart_disease_data
        train, test, meta = load_heart_disease_data(
            num_clients=cfg["num_clients"], seed=seed
        )
    elif loader_name == "ptbxl":
        from data.ptbxl_loader import load_ptbxl_data
        train, test, meta = load_ptbxl_data(
            num_clients=cfg["num_clients"], seed=seed
        )
    else:
        raise ValueError(f"Unknown loader: {loader_name}")

    _data_cache[cache_key] = (train, test, meta)
    return (
        {cid: (X.copy(), y.copy()) for cid, (X, y) in train.items()},
        test,
        meta,
    )


def _inject_noise(
    train_data: Dict[int, Tuple[np.ndarray, np.ndarray]],
    noise_type: str,
    noise_level: float,
    seed: int,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Inject noise into training data for data quality stress testing.

    noise_type: "label", "feature", "missing"
    noise_level: fraction of data to corrupt (0.0 - 1.0)
    """
    rng = np.random.RandomState(seed)
    noisy = {}
    for cid, (X, y) in train_data.items():
        X_n = X.copy()
        y_n = y.copy()
        n = len(y_n)
        n_corrupt = max(1, int(n * noise_level))
        corrupt_idx = rng.choice(n, size=n_corrupt, replace=False)

        if noise_type == "label":
            # Random label flip
            num_classes = len(np.unique(y_n))
            for i in corrupt_idx:
                new_label = rng.randint(0, num_classes)
                while new_label == y_n[i] and num_classes > 1:
                    new_label = rng.randint(0, num_classes)
                y_n[i] = new_label

        elif noise_type == "feature":
            # Additive Gaussian noise
            noise = rng.normal(0, noise_level, size=X_n[corrupt_idx].shape)
            X_n[corrupt_idx] += noise.astype(X_n.dtype)

        elif noise_type == "missing":
            # Zero-fill random features
            n_features = X_n.shape[1]
            for i in corrupt_idx:
                n_missing = max(1, rng.randint(1, max(2, n_features // 3)))
                feat_idx = rng.choice(n_features, size=n_missing, replace=False)
                X_n[i, feat_idx] = 0.0

        noisy[cid] = (X_n, y_n)
    return noisy


# ============================================================================
# PHASE 2A: GOVERNANCE OVERHEAD BENCHMARK
# ============================================================================

def _build_phase_2a(quick: bool) -> List[Dict]:
    """
    Governance overhead: measure per-round time with/without governance hooks.
    Dataset: Cardiovascular (fast, 70K samples, 5 clients).
    """
    algos = ALGOS_CORE if not quick else ["FedAvg", "HPFL"]
    seeds = SEEDS_3 if not quick else [42]
    conditions = ["no_governance", "with_governance"]
    exps = []
    for algo in algos:
        for cond in conditions:
            for seed in seeds:
                exps.append({
                    "key": f"2A_{algo}_{cond}_s{seed}",
                    "phase": "2A",
                    "algo": algo,
                    "seed": seed,
                    "governance": cond == "with_governance",
                    "dataset": "Cardiovascular",
                })
    return exps


def _run_phase_2a(exp: Dict, quick: bool) -> Dict:
    """Run governance overhead experiment."""
    from terminal.fl_trainer import FederatedTrainer
    from terminal.training.models import HealthcareMLP

    cfg = TABULAR_DATASETS[exp["dataset"]]
    nr = 10 if quick else 30
    seed = exp["seed"]
    algo = exp["algo"]

    train, test, meta = _load_tabular(exp["dataset"], seed)

    dp_enabled = False
    dp_epsilon = 10.0

    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm=algo,
        local_epochs=3,
        batch_size=64,
        learning_rate=0.005,
        mu=0.1,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_clip_norm=1.0,
        seed=seed,
        external_data=train,
        external_test_data=test,
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
    )

    # Optional governance wrapper
    gov_ctx = None
    round_times_gov = []
    if exp["governance"]:
        from core.models import DataCategory, PermitPurpose
        from governance.permit_training import PermitAwareTrainingContext

        gov_ctx = PermitAwareTrainingContext(
            permit_id=f"GOV-BENCH-{algo}-s{seed}",
            purpose=PermitPurpose.SCIENTIFIC_RESEARCH,
            data_categories=[DataCategory.EHR, DataCategory.LAB_RESULTS],
            privacy_budget_total=100.0,
            max_rounds=nr + 5,
            client_ids=[f"client_{i}" for i in range(cfg["num_clients"])],
            audit_output_dir=str(OUTPUT_DIR / "gov_audit_logs"),
        )
        gov_ctx.start_session()

    history = []
    t0_total = time.time()

    for r in range(nr):
        if _shutdown:
            break

        t_round_start = time.time()

        # Governance pre-round validation
        if gov_ctx is not None:
            eps_cost = 0.01 if dp_enabled else 0.0
            ok, reason = gov_ctx.validate_round(r, eps_cost)
            if not ok:
                log(f"    Governance blocked round {r}: {reason}")
                break

        res = trainer.train_round(r)

        # Governance post-round logging
        if gov_ctx is not None:
            gov_ctx.log_round_completion(res, epsilon_spent=0.01 if dp_enabled else 0.0)

        t_round_end = time.time()
        round_time = t_round_end - t_round_start
        round_times_gov.append(round_time)

        acc = getattr(res, "global_acc", 0)
        history.append({"round": r + 1, "accuracy": round(acc, 4), "round_time_s": round(round_time, 4)})

    if gov_ctx is not None:
        gov_ctx.end_session(
            total_rounds=len(history),
            final_metrics={"accuracy": history[-1]["accuracy"] if history else 0},
        )

    elapsed = round(time.time() - t0_total, 1)
    final_acc = history[-1]["accuracy"] if history else 0
    mean_round_time = float(np.mean(round_times_gov)) if round_times_gov else 0

    return {
        "algorithm": algo,
        "dataset": exp["dataset"],
        "governance": exp["governance"],
        "seed": seed,
        "num_rounds": nr,
        "actual_rounds": len(history),
        "final_accuracy": round(final_acc, 4),
        "mean_round_time_s": round(mean_round_time, 4),
        "std_round_time_s": round(float(np.std(round_times_gov)), 4) if round_times_gov else 0,
        "total_runtime_s": elapsed,
        "round_times": [round(t, 4) for t in round_times_gov],
        "history": history,
    }


# ============================================================================
# PHASE 1B: DATA QUALITY STRESS TESTING
# ============================================================================

def _build_phase_1b(quick: bool) -> List[Dict]:
    """
    Data quality stress: label noise, feature noise, missing features.
    Datasets: Cardiovascular (large), Heart_Disease (small).
    """
    if quick:
        datasets = ["Cardiovascular"]
        algos = ["FedAvg", "HPFL"]
        seeds = [42]
        noise_configs = [
            {"noise_type": "label", "noise_level": 0.1},
            {"noise_type": "label", "noise_level": 0.3},
            {"noise_type": "feature", "noise_level": 0.2},
            {"noise_type": "missing", "noise_level": 0.2},
        ]
    else:
        datasets = ["Cardiovascular", "Heart_Disease"]
        algos = ALGOS_CORE
        seeds = SEEDS_3
        noise_configs = [
            # Baselines (no noise — but we can reference existing results)
            {"noise_type": "label", "noise_level": 0.05},
            {"noise_type": "label", "noise_level": 0.1},
            {"noise_type": "label", "noise_level": 0.2},
            {"noise_type": "label", "noise_level": 0.3},
            {"noise_type": "feature", "noise_level": 0.1},
            {"noise_type": "feature", "noise_level": 0.2},
            {"noise_type": "feature", "noise_level": 0.5},
            {"noise_type": "missing", "noise_level": 0.1},
            {"noise_type": "missing", "noise_level": 0.2},
            {"noise_type": "missing", "noise_level": 0.3},
        ]

    exps = []
    for ds in datasets:
        for algo in algos:
            for nc in noise_configs:
                for seed in seeds:
                    nt = nc["noise_type"]
                    nl = str(nc["noise_level"]).replace(".", "p")
                    exps.append({
                        "key": f"1B_{ds}_{algo}_{nt}_{nl}_s{seed}",
                        "phase": "1B",
                        "algo": algo,
                        "seed": seed,
                        "dataset": ds,
                        "noise_type": nc["noise_type"],
                        "noise_level": nc["noise_level"],
                    })
    return exps


def _run_phase_1b(exp: Dict, quick: bool) -> Dict:
    """Run data quality stress experiment."""
    from terminal.fl_trainer import FederatedTrainer

    cfg = TABULAR_DATASETS[exp["dataset"]]
    nr = 10 if quick else 30
    seed = exp["seed"]
    algo = exp["algo"]

    train, test, meta = _load_tabular(exp["dataset"], seed)

    # Inject noise
    train = _inject_noise(train, exp["noise_type"], exp["noise_level"], seed)

    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm=algo,
        local_epochs=3,
        batch_size=64,
        learning_rate=0.005,
        mu=0.1,
        dp_enabled=False,
        dp_epsilon=10.0,
        dp_clip_norm=1.0,
        seed=seed,
        external_data=train,
        external_test_data=test,
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
    )

    history = []
    best_acc = 0.0
    t0 = time.time()

    for r in range(nr):
        if _shutdown:
            break
        res = trainer.train_round(r)
        acc = getattr(res, "global_acc", 0)
        history.append({"round": r + 1, "accuracy": round(acc, 4)})
        if acc > best_acc:
            best_acc = acc

    elapsed = round(time.time() - t0, 1)
    final_acc = history[-1]["accuracy"] if history else 0

    return {
        "algorithm": algo,
        "dataset": exp["dataset"],
        "noise_type": exp["noise_type"],
        "noise_level": exp["noise_level"],
        "seed": seed,
        "num_rounds": nr,
        "actual_rounds": len(history),
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "runtime_seconds": elapsed,
    }


# ============================================================================
# PHASE 5B: HPFL SPLIT-POINT ABLATION (IMAGING)
# ============================================================================

def _build_phase_5b(quick: bool) -> List[Dict]:
    """
    HPFL split-point ablation on imaging datasets.
    depth=1: classifier only (default)
    depth=2: layer4 + classifier
    depth=3: layer3 + layer4 + classifier
    """
    if quick:
        datasets = ["Brain_Tumor"]
        seeds = [42]
    else:
        datasets = list(IMAGING_DATASETS.keys())
        seeds = SEEDS_3

    depths = [1, 2, 3]
    exps = []
    for ds in datasets:
        for depth in depths:
            for seed in seeds:
                exps.append({
                    "key": f"5B_{ds}_HPFL_d{depth}_s{seed}",
                    "phase": "5B",
                    "algo": "HPFL",
                    "seed": seed,
                    "dataset": ds,
                    "split_depth": depth,
                })
    return exps


def _patch_hpfl_split_depth(trainer, depth: int):
    """
    Monkey-patch HPFL classifier boundary to include deeper layers.

    depth=1: classifier only (default, matches 'classifier' pattern)
    depth=2: layer4 + classifier
    depth=3: layer3 + layer4 + classifier

    Only applies to ResNet-based models.
    """
    # Start from default classifier-only names
    names: Set[str] = set(trainer._hpfl_classifier_names)

    if depth >= 2:
        for n, _ in trainer.global_model.named_parameters():
            if n.startswith("layer4."):
                names.add(n)
    if depth >= 3:
        for n, _ in trainer.global_model.named_parameters():
            if n.startswith("layer3."):
                names.add(n)

    # Update trainer state
    trainer._hpfl_classifier_names = names
    # Reinitialize client classifiers with the expanded set
    trainer.client_classifiers = {
        i: {n: trainer.global_model.state_dict()[n].clone() for n in names}
        for i in range(trainer.num_clients)
    }


def _run_phase_5b(exp: Dict, quick: bool) -> Dict:
    """Run HPFL split-point ablation experiment."""
    from terminal.training.federated_image import ImageFederatedTrainer

    ds_name = exp["dataset"]
    ds_cfg = IMAGING_DATASETS[ds_name]
    seed = exp["seed"]
    depth = exp["split_depth"]
    nr = 5 if quick else 15

    lr = IMAGING_LR_OVERRIDES.get(ds_name, {}).get("learning_rate", 0.0005)

    trainer = ImageFederatedTrainer(
        data_dir=ds_cfg["data_dir"],
        num_clients=ds_cfg["num_clients"],
        algorithm="HPFL",
        local_epochs=3,
        batch_size=32,
        learning_rate=lr,
        is_iid=False,
        alpha=0.5,
        mu=0.1,
        dp_enabled=False,
        dp_epsilon=10.0,
        dp_clip_norm=1.0,
        seed=seed,
        model_type="resnet18",
        freeze_backbone=False,
    )
    trainer.num_rounds = nr

    # Patch HPFL split point
    _patch_hpfl_split_depth(trainer, depth)
    n_personal_params = sum(
        p.numel() for n, p in trainer.global_model.named_parameters()
        if n in trainer._hpfl_classifier_names
    )
    n_total_params = sum(p.numel() for p in trainer.global_model.parameters())

    log(f"    Split depth={depth}: {len(trainer._hpfl_classifier_names)} personal param groups, "
        f"{n_personal_params:,}/{n_total_params:,} params ({100*n_personal_params/n_total_params:.1f}%)")

    history = []
    best_acc = 0.0
    t0 = time.time()

    for r in range(nr):
        if _shutdown:
            break
        res = trainer.train_round(r)
        acc = getattr(res, "global_acc", 0)
        history.append({"round": r + 1, "accuracy": round(acc, 4)})
        if acc > best_acc:
            best_acc = acc

    elapsed = round(time.time() - t0, 1)
    final_acc = history[-1]["accuracy"] if history else 0

    _cleanup_gpu()

    return {
        "algorithm": "HPFL",
        "dataset": ds_name,
        "split_depth": depth,
        "n_personal_params": n_personal_params,
        "n_total_params": n_total_params,
        "personal_frac": round(n_personal_params / n_total_params, 4),
        "seed": seed,
        "num_rounds": nr,
        "actual_rounds": len(history),
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "runtime_seconds": elapsed,
    }


# ============================================================================
# PHASE 4A: EFFICIENTNET-B0 IMAGING
# ============================================================================

def _build_phase_4a(quick: bool) -> List[Dict]:
    """EfficientNet-B0 on 3 imaging datasets, 3 algos, 3 seeds."""
    if quick:
        datasets = ["Brain_Tumor"]
        algos = ["FedAvg"]
        seeds = [42]
    else:
        datasets = list(IMAGING_DATASETS.keys())
        algos = ALGOS_CORE
        seeds = SEEDS_3

    exps = []
    for ds in datasets:
        for algo in algos:
            for seed in seeds:
                exps.append({
                    "key": f"4A_{ds}_{algo}_EffB0_s{seed}",
                    "phase": "4A",
                    "algo": algo,
                    "seed": seed,
                    "dataset": ds,
                })
    return exps


def _run_phase_4a(exp: Dict, quick: bool) -> Dict:
    """Run EfficientNet-B0 imaging experiment."""
    from terminal.training.federated_image import ImageFederatedTrainer

    ds_name = exp["dataset"]
    ds_cfg = IMAGING_DATASETS[ds_name]
    seed = exp["seed"]
    algo = exp["algo"]
    nr = 5 if quick else 15

    lr = IMAGING_LR_OVERRIDES.get(ds_name, {}).get("learning_rate", 0.0005)

    # Create trainer with resnet18 first (to load data), then swap model
    trainer = ImageFederatedTrainer(
        data_dir=ds_cfg["data_dir"],
        num_clients=ds_cfg["num_clients"],
        algorithm=algo,
        local_epochs=3,
        batch_size=32,
        learning_rate=lr,
        is_iid=False,
        alpha=0.5,
        mu=0.1,
        dp_enabled=False,
        dp_epsilon=10.0,
        dp_clip_norm=1.0,
        seed=seed,
        model_type="resnet18",  # placeholder, will swap
        freeze_backbone=False,
        img_size=224,
    )
    trainer.num_rounds = nr

    # Swap model to EfficientNet-B0
    num_classes = trainer.num_classes
    new_model = HealthcareEfficientNet(num_classes=num_classes).to(trainer.device)
    trainer.global_model = new_model

    # Re-initialize algorithm state that depends on model
    if algo == "HPFL":
        trainer._hpfl_classifier_names = trainer._identify_classifier_params()
        trainer.client_classifiers = {
            i: {n: trainer.global_model.state_dict()[n].clone()
                for n in trainer._hpfl_classifier_names}
            for i in range(trainer.num_clients)
        }
    elif algo == "Ditto":
        trainer.personalized_models = {
            i: deepcopy(trainer.global_model)
            for i in range(trainer.num_clients)
        }

    n_params = sum(p.numel() for p in trainer.global_model.parameters())
    log(f"    EfficientNet-B0: {n_params:,} params, {num_classes} classes")

    history = []
    best_acc = 0.0
    t0 = time.time()

    for r in range(nr):
        if _shutdown:
            break
        res = trainer.train_round(r)
        acc = getattr(res, "global_acc", 0)
        history.append({"round": r + 1, "accuracy": round(acc, 4)})
        if acc > best_acc:
            best_acc = acc

    elapsed = round(time.time() - t0, 1)
    final_acc = history[-1]["accuracy"] if history else 0

    _cleanup_gpu()

    return {
        "algorithm": algo,
        "dataset": ds_name,
        "model": "EfficientNet-B0",
        "num_params": n_params,
        "seed": seed,
        "num_rounds": nr,
        "actual_rounds": len(history),
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "runtime_seconds": elapsed,
    }


# ============================================================================
# PHASE 3A: TABNET FEDERATED (TABULAR)
# ============================================================================

def _build_phase_3a(quick: bool) -> List[Dict]:
    """TabNet on 3 tabular datasets, 3 algos, 5 seeds."""
    if quick:
        datasets = ["Cardiovascular"]
        algos = ["FedAvg"]
        seeds = [42]
    else:
        datasets = list(TABULAR_DATASETS.keys())
        algos = ALGOS_CORE
        seeds = SEEDS_5

    exps = []
    for ds in datasets:
        for algo in algos:
            for seed in seeds:
                exps.append({
                    "key": f"3A_{ds}_{algo}_TabNet_s{seed}",
                    "phase": "3A",
                    "algo": algo,
                    "seed": seed,
                    "dataset": ds,
                })
    return exps


def _run_phase_3a(exp: Dict, quick: bool) -> Dict:
    """Run TabNet federated experiment."""
    from terminal.fl_trainer import FederatedTrainer
    import terminal.training.federated as _fed_module

    ds_name = exp["dataset"]
    cfg = TABULAR_DATASETS[ds_name]
    seed = exp["seed"]
    algo = exp["algo"]
    nr = 10 if quick else 30

    train, test, meta = _load_tabular(ds_name, seed)

    trainer = FederatedTrainer(
        num_clients=cfg["num_clients"],
        algorithm=algo,
        local_epochs=3,
        batch_size=64,
        learning_rate=0.005,
        mu=0.1,
        dp_enabled=False,
        dp_epsilon=10.0,
        dp_clip_norm=1.0,
        seed=seed,
        external_data=train,
        external_test_data=test,
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
    )

    # Swap model to TabNet
    new_model = HealthcareTabNet(
        input_dim=cfg["input_dim"],
        num_classes=cfg["num_classes"],
        n_d=64,
        n_a=64,
        n_steps=3,
    ).to(trainer.device)
    trainer.global_model = new_model

    # Re-initialize algorithm state that depends on model
    if algo == "HPFL":
        trainer._hpfl_classifier_names = trainer._identify_classifier_params()
        trainer.client_classifiers = {
            i: {n: trainer.global_model.state_dict()[n].clone()
                for n in trainer._hpfl_classifier_names}
            for i in range(trainer.num_clients)
        }
    elif algo == "Ditto":
        trainer.personalized_models = {
            i: deepcopy(trainer.global_model)
            for i in range(trainer.num_clients)
        }
    elif algo == "SCAFFOLD":
        trainer.server_control = {
            name: torch.zeros_like(param)
            for name, param in trainer.global_model.named_parameters()
        }
        trainer.client_controls = {
            i: {name: torch.zeros_like(param)
                for name, param in trainer.global_model.named_parameters()}
            for i in range(trainer.num_clients)
        }

    n_params = sum(p.numel() for p in trainer.global_model.parameters())
    log(f"    TabNet: {n_params:,} params, {cfg['num_classes']} classes, input_dim={cfg['input_dim']}")

    history = []
    best_acc = 0.0
    t0 = time.time()

    for r in range(nr):
        if _shutdown:
            break
        res = trainer.train_round(r)
        acc = getattr(res, "global_acc", 0)
        history.append({"round": r + 1, "accuracy": round(acc, 4)})
        if acc > best_acc:
            best_acc = acc

    elapsed = round(time.time() - t0, 1)
    final_acc = history[-1]["accuracy"] if history else 0

    return {
        "algorithm": algo,
        "dataset": ds_name,
        "model": "TabNet",
        "num_params": n_params,
        "seed": seed,
        "num_rounds": nr,
        "actual_rounds": len(history),
        "final_accuracy": round(final_acc, 4),
        "best_accuracy": round(best_acc, 4),
        "runtime_seconds": elapsed,
    }


# ============================================================================
# EXPERIMENT BUILDERS & RUNNERS — DISPATCH TABLE
# ============================================================================

PHASE_BUILDERS = {
    "2A": _build_phase_2a,
    "1B": _build_phase_1b,
    "5B": _build_phase_5b,
    "4A": _build_phase_4a,
    "3A": _build_phase_3a,
}

PHASE_RUNNERS = {
    "2A": _run_phase_2a,
    "1B": _run_phase_1b,
    "5B": _run_phase_5b,
    "4A": _run_phase_4a,
    "3A": _run_phase_3a,
}

PHASE_ORDER = ["2A", "1B", "5B", "4A", "3A"]

PHASE_DESCRIPTIONS = {
    "2A": "Governance overhead benchmark (Cardiovascular, ±governance hooks)",
    "1B": "Data quality stress (label/feature/missing noise)",
    "5B": "HPFL split-point ablation (imaging, depth 1-3)",
    "4A": "EfficientNet-B0 imaging (architecture generality)",
    "3A": "TabNet federated tabular (architecture generality)",
}


def build_all_experiments(quick: bool, start_phase: Optional[str] = None) -> List[Dict]:
    """Build complete experiment list ordered by phase execution time."""
    exps = []
    started = start_phase is None
    for phase in PHASE_ORDER:
        if not started:
            if phase == start_phase:
                started = True
            else:
                continue
        builder = PHASE_BUILDERS[phase]
        exps.extend(builder(quick))
    return exps


# ============================================================================
# SUMMARY PRINTERS
# ============================================================================

def print_2a_summary(completed: Dict):
    """Summary: governance overhead."""
    results = {k: v for k, v in completed.items() if k.startswith("2A_") and "error" not in v}
    if not results:
        return
    log("\n--- Phase 2A: Governance Overhead ---")
    log(f"  {'Algorithm':<10} | {'No-Gov (s/round)':>18} | {'With-Gov (s/round)':>20} | {'Overhead':>10}")
    log("  " + "-" * 66)

    for algo in ALGOS_CORE:
        no_gov = [v["mean_round_time_s"] for v in results.values()
                  if v.get("algorithm") == algo and not v.get("governance")]
        with_gov = [v["mean_round_time_s"] for v in results.values()
                    if v.get("algorithm") == algo and v.get("governance")]
        if no_gov and with_gov:
            ng = np.mean(no_gov)
            wg = np.mean(with_gov)
            overhead_pct = ((wg - ng) / ng * 100) if ng > 0 else 0
            log(f"  {algo:<10} | {ng:>17.4f}s | {wg:>19.4f}s | {overhead_pct:>+9.1f}%")


def print_1b_summary(completed: Dict):
    """Summary: data quality stress."""
    results = {k: v for k, v in completed.items() if k.startswith("1B_") and "error" not in v}
    if not results:
        return
    log("\n--- Phase 1B: Data Quality Stress ---")

    for ds in ["Cardiovascular", "Heart_Disease"]:
        ds_results = {k: v for k, v in results.items() if v.get("dataset") == ds}
        if not ds_results:
            continue
        log(f"\n  Dataset: {ds}")
        log(f"  {'Noise':<10} | {'Level':>5} | {'FedAvg':>8} | {'Ditto':>8} | {'HPFL':>8}")
        log("  " + "-" * 52)

        noise_types = sorted(set(v.get("noise_type", "") for v in ds_results.values()))
        for nt in noise_types:
            levels = sorted(set(v.get("noise_level", 0) for v in ds_results.values() if v.get("noise_type") == nt))
            for nl in levels:
                row = {}
                for algo in ALGOS_CORE:
                    accs = [v["best_accuracy"] * 100 for v in ds_results.values()
                            if v.get("algorithm") == algo and v.get("noise_type") == nt
                            and abs(v.get("noise_level", 0) - nl) < 0.001]
                    row[algo] = f"{np.mean(accs):>6.1f}%" if accs else "    N/A"
                log(f"  {nt:<10} | {nl:>5.2f} | {row.get('FedAvg', '    N/A'):>8} | "
                    f"{row.get('Ditto', '    N/A'):>8} | {row.get('HPFL', '    N/A'):>8}")


def print_5b_summary(completed: Dict):
    """Summary: HPFL split-point ablation."""
    results = {k: v for k, v in completed.items() if k.startswith("5B_") and "error" not in v}
    if not results:
        return
    log("\n--- Phase 5B: HPFL Split-Point Ablation ---")
    log(f"  {'Dataset':<14} | {'Depth':>5} | {'Personal%':>9} | {'Best Acc':>10}")
    log("  " + "-" * 48)

    for ds in IMAGING_DATASETS:
        for depth in [1, 2, 3]:
            accs = [v["best_accuracy"] * 100 for v in results.values()
                    if v.get("dataset") == ds and v.get("split_depth") == depth]
            fracs = [v.get("personal_frac", 0) * 100 for v in results.values()
                     if v.get("dataset") == ds and v.get("split_depth") == depth]
            if accs:
                mean_frac = np.mean(fracs)
                log(f"  {ds:<14} | {depth:>5} | {mean_frac:>8.1f}% | {np.mean(accs):>6.1f}±{np.std(accs):.1f}%")


def print_4a_summary(completed: Dict):
    """Summary: EfficientNet-B0 imaging."""
    results = {k: v for k, v in completed.items() if k.startswith("4A_") and "error" not in v}
    if not results:
        return
    log("\n--- Phase 4A: EfficientNet-B0 Imaging ---")
    log(f"  {'Dataset':<14} | {'Algorithm':<10} | {'Best Acc':>10} | {'Seeds':>5}")
    log("  " + "-" * 48)

    for ds in IMAGING_DATASETS:
        for algo in ALGOS_CORE:
            accs = [v["best_accuracy"] * 100 for v in results.values()
                    if v.get("dataset") == ds and v.get("algorithm") == algo]
            if accs:
                log(f"  {ds:<14} | {algo:<10} | {np.mean(accs):>6.1f}±{np.std(accs):.1f}% | {len(accs):>5}")


def print_3a_summary(completed: Dict):
    """Summary: TabNet federated tabular."""
    results = {k: v for k, v in completed.items() if k.startswith("3A_") and "error" not in v}
    if not results:
        return
    log("\n--- Phase 3A: TabNet Federated (Tabular) ---")
    log(f"  {'Dataset':<16} | {'Algorithm':<10} | {'Best Acc':>10} | {'Seeds':>5}")
    log("  " + "-" * 50)

    for ds in TABULAR_DATASETS:
        for algo in ALGOS_CORE:
            accs = [v["best_accuracy"] * 100 for v in results.values()
                    if v.get("dataset") == ds and v.get("algorithm") == algo]
            if accs:
                log(f"  {ds:<16} | {algo:<10} | {np.mean(accs):>6.1f}±{np.std(accs):.1f}% | {len(accs):>5}")


PHASE_SUMMARIES = {
    "2A": print_2a_summary,
    "1B": print_1b_summary,
    "5B": print_5b_summary,
    "4A": print_4a_summary,
    "3A": print_3a_summary,
}


# ============================================================================
# MAIN
# ============================================================================

def main():
    global _checkpoint_data

    # Parse arguments
    if "--quick" in sys.argv:
        quick = True
    elif "--full" in sys.argv:
        quick = False
    else:
        print("Usage: python -m benchmarks.run_reviewer_cascade_2 --quick|--full [--phase PHASE]")
        print("  Phases: 2A, 1B, 5B, 4A, 3A")
        sys.exit(1)

    start_phase = None
    if "--phase" in sys.argv:
        idx = sys.argv.index("--phase")
        if idx + 1 < len(sys.argv):
            start_phase = sys.argv[idx + 1]
            if start_phase not in PHASE_ORDER:
                print(f"Unknown phase: {start_phase}. Valid: {PHASE_ORDER}")
                sys.exit(1)

    mode_label = "QUICK" if quick else "FULL"

    # Build experiments
    experiments = build_all_experiments(quick=quick, start_phase=start_phase)
    total = len(experiments)

    # Verify no duplicate keys
    keys = [e["key"] for e in experiments]
    dupes = [k for k in set(keys) if keys.count(k) > 1]
    if dupes:
        log(f"FATAL: {len(dupes)} duplicate experiment keys: {dupes[:5]}")
        sys.exit(1)

    # Count per phase
    phase_counts = {}
    for e in experiments:
        phase_counts[e["phase"]] = phase_counts.get(e["phase"], 0) + 1

    log("=" * 70)
    log(f"REVIEWER CASCADE 2 — {mode_label} MODE")
    log(f"5 phases: {' → '.join(PHASE_ORDER)}")
    log(f"Total experiments: {total}")
    for ph in PHASE_ORDER:
        if ph in phase_counts:
            log(f"  Phase {ph}: {phase_counts[ph]:>3} experiments — {PHASE_DESCRIPTIONS[ph]}")
    if start_phase:
        log(f"Starting from phase: {start_phase}")
    log("=" * 70)

    # Load or init checkpoint
    _checkpoint_data = load_checkpoint() or {
        "completed": {},
        "metadata": {
            "total_experiments": total,
            "mode": mode_label,
            "start_time": datetime.now().isoformat(),
        },
    }
    completed = _checkpoint_data.setdefault("completed", {})
    _checkpoint_data["metadata"]["total_experiments"] = total
    done = sum(1 for k in completed if any(e["key"] == k for e in experiments))
    if done > 0:
        log(f"\nResumed: {done} previously completed")

    t_start = time.time()
    current_phase = ""
    exp_counter = done

    for exp in experiments:
        if _shutdown:
            break
        key = exp["key"]
        if key in completed:
            continue

        # Phase transition
        if exp["phase"] != current_phase:
            # Print summary of previous phase
            if current_phase and current_phase in PHASE_SUMMARIES:
                PHASE_SUMMARIES[current_phase](completed)
            current_phase = exp["phase"]
            ph_total = phase_counts.get(current_phase, 0)
            ph_done = sum(
                1 for k in completed
                if any(e["key"] == k and e["phase"] == current_phase for e in experiments)
            )
            log(f"\n{'=' * 70}")
            log(f"PHASE {current_phase}: {PHASE_DESCRIPTIONS[current_phase]}")
            log(f"  ({ph_done}/{ph_total} done)")
            log(f"{'=' * 70}")

        exp_counter += 1
        log(f"\n[{exp_counter}/{total}] {key} ...")

        try:
            runner = PHASE_RUNNERS[exp["phase"]]
            result = runner(exp, quick)
            result["key"] = key
            result["phase"] = exp["phase"]
            completed[key] = result
            save_checkpoint()

            # Log result line
            acc_str = f"best={result.get('best_accuracy', result.get('final_accuracy', 0))*100:.1f}%"
            rt_str = f"{result.get('runtime_seconds', result.get('total_runtime_s', 0)):.0f}s"
            log(f"  -> {acc_str}  ({rt_str})")

        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            completed[key] = {"key": key, "phase": exp["phase"], "error": str(e)}
            save_checkpoint()

        _cleanup_gpu()

    # Final save
    elapsed = time.time() - t_start
    _checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    _checkpoint_data["metadata"]["elapsed_seconds"] = round(elapsed, 1)
    save_checkpoint()

    # ============================================================
    # FINAL SUMMARY
    # ============================================================
    log("\n" + "=" * 70)
    total_done = sum(1 for k in completed if any(e["key"] == k for e in experiments))
    log(f"REVIEWER CASCADE 2 {mode_label} COMPLETE: {total_done}/{total} in {elapsed:.0f}s")
    log("=" * 70)

    errors = [k for k, v in completed.items() if "error" in v]
    if errors:
        log(f"\nERRORS ({len(errors)}):")
        for k in errors[:10]:
            log(f"  {k}: {completed[k]['error']}")

    for ph in PHASE_ORDER:
        if ph in PHASE_SUMMARIES:
            PHASE_SUMMARIES[ph](completed)

    log(f"\nCheckpoint: {OUTPUT_DIR / CHECKPOINT_FILE}")


if __name__ == "__main__":
    main()
