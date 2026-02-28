#!/usr/bin/env python3
"""
FL-EHDS Imaging — DP + Opt-Out Combined (EHDS Realistic Scenario).

Tests the joint effect of Differential Privacy (eps=5) AND GDPR Article 71
opt-out on imaging datasets. This is the realistic EHDS deployment scenario
where hospitals apply DP noise AND citizens exercise their right to data erasure.

Experiments: 3 algos x 3 datasets x 5 rates x 3 seeds = 135 experiments
Estimated runtime: ~5-7 hours on RTX 4090, ~18-24 hours on T4

Usage:
    cd fl-ehds-framework
    python -m benchmarks.run_imaging_dp_optout [--download] [--fresh] [--dataset chest_xray]

Output: benchmarks/checkpoint_imaging_dp_optout.json

Author: Fabio Liberti
"""

import sys
import os
import json
import time
import shutil
import signal
import tempfile
import argparse
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional, Any

import numpy as np

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

import torch

from terminal.fl_trainer import (
    ImageFederatedTrainer,
    _detect_device,
)

# ======================================================================
# Configuration
# ======================================================================

ALGORITHMS = ["FedAvg", "Ditto", "HPFL"]

OPTOUT_RATES = [0.0, 0.05, 0.10, 0.20, 0.30]

# Fixed DP configuration (epsilon=5 = strong privacy, balanced tradeoff)
DP_EPSILON = 5.0
DP_CLIP_NORM = 1.0

SEEDS = [42, 123, 456]

IMAGING_DATASETS = {
    "chest_xray": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "chest_xray"),
        "num_classes": 2,
        "short": "CX",
    },
    "Brain_Tumor": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Brain_Tumor"),
        "num_classes": 4,
        "short": "BT",
    },
    "Skin_Cancer": {
        "data_dir": str(FRAMEWORK_DIR / "data" / "Skin Cancer"),
        "num_classes": 2,
        "short": "SC",
    },
}

OUTPUT_DIR = FRAMEWORK_DIR / "benchmarks"
CHECKPOINT_FILE = "checkpoint_imaging_dp_optout.json"
LOG_FILE = "experiment_imaging_dp_optout.log"
TRAINER_STATE_FILE = ".trainer_state_dp_optout.pt"

IMAGING_CONFIG = dict(
    num_clients=5,
    num_rounds=20,
    local_epochs=2,
    batch_size=32,
    learning_rate=0.001,
    model_type="resnet18",
    is_iid=False,
    alpha=0.5,
    freeze_backbone=False,
    freeze_level=2,
    use_fedbn=True,
    use_class_weights=True,
    use_amp=True,
    mu=0.1,
)

EARLY_STOPPING = dict(
    enabled=True,
    patience=4,
    min_delta=0.003,
    min_rounds=8,
    metric="accuracy",
)

DATASET_OVERRIDES = {
    "Brain_Tumor": {"learning_rate": 0.0005},
}


# ======================================================================
# Build experiment list
# ======================================================================

def build_experiments(datasets=None):
    experiments = []
    ds_list = datasets if datasets else list(IMAGING_DATASETS.keys())
    for rate in OPTOUT_RATES:
        for ds_name in ds_list:
            for algo in ALGORITHMS:
                for seed in SEEDS:
                    experiments.append((ds_name, algo, rate, seed))
    return experiments


def experiment_key(ds_name, algo, rate, seed):
    return f"{ds_name}_{algo}_dp{int(DP_EPSILON)}_opt{int(rate*100)}_s{seed}"


# ======================================================================
# Logging
# ======================================================================

_log_file = None

def log(msg: str, also_print: bool = True):
    ts = datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}"
    if also_print:
        print(line, flush=True)
    if _log_file:
        try:
            _log_file.write(line + "\n")
            _log_file.flush()
        except Exception:
            pass


# ======================================================================
# Checkpoint — atomic save with backup
# ======================================================================

def save_checkpoint(data: Dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak_path = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    data["metadata"]["last_save"] = datetime.now().isoformat()

    fd, tmp_path = tempfile.mkstemp(
        dir=str(OUTPUT_DIR), prefix=".ckpt_dpopt_", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        if path.exists():
            shutil.copy2(str(path), str(bak_path))
        os.replace(tmp_path, str(path))
        log(f"  [SAVED] checkpoint: {len(data.get('completed', {}))} experiments", also_print=False)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_checkpoint() -> Optional[Dict]:
    path = OUTPUT_DIR / CHECKPOINT_FILE
    bak_path = OUTPUT_DIR / (CHECKPOINT_FILE + ".bak")
    for p in [path, bak_path]:
        if p.exists():
            try:
                with open(p, "r") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                log(f"  WARNING: corrupt checkpoint {p.name}, trying backup...")
                continue
    return None


# ======================================================================
# GPU cleanup
# ======================================================================

def _cleanup_gpu():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass
    import gc
    gc.collect()


# ======================================================================
# Early Stopping
# ======================================================================

class EarlyStoppingMonitor:
    def __init__(self, patience=4, min_delta=0.003, min_rounds=8, metric="accuracy"):
        self.patience = patience
        self.min_delta = min_delta
        self.min_rounds = min_rounds
        self.metric = metric
        self.best_value = -float('inf')
        self.best_round = 0
        self.counter = 0

    def check(self, round_num, metrics):
        value = metrics.get(self.metric, 0)
        if value > self.best_value + self.min_delta:
            self.best_value = value
            self.best_round = round_num
            self.counter = 0
        else:
            self.counter += 1
        if round_num < self.min_rounds:
            return False
        return self.counter >= self.patience


# ======================================================================
# Per-client evaluation & fairness
# ======================================================================

def _evaluate_per_client(trainer) -> Dict[str, float]:
    model = trainer.global_model
    model.eval()
    per_client = {}

    is_hpfl = trainer.algorithm == "HPFL"
    if is_hpfl and hasattr(trainer, '_hpfl_classifier_names'):
        saved_cls = {n: p.data.clone() for n, p in model.named_parameters()
                     if n in trainer._hpfl_classifier_names}
    else:
        is_hpfl = False

    with torch.no_grad():
        for cid in range(trainer.num_clients):
            if is_hpfl:
                for n, p in model.named_parameters():
                    if n in trainer._hpfl_classifier_names:
                        p.data.copy_(trainer.client_classifiers[cid][n])

            X, y = trainer.client_test_data[cid]
            X_t = torch.FloatTensor(X).to(trainer.device) if isinstance(X, np.ndarray) else X.to(trainer.device)
            y_t = torch.LongTensor(y).to(trainer.device) if isinstance(y, np.ndarray) else y.to(trainer.device)
            correct = total = 0
            bs = 64
            for i in range(0, len(y_t), bs):
                out = model(X_t[i:i+bs])
                preds = out.argmax(dim=1)
                correct += (preds == y_t[i:i+bs]).sum().item()
                total += len(y_t[i:i+bs])
            per_client[str(cid)] = correct / total if total > 0 else 0.0

    if is_hpfl:
        for n, p in model.named_parameters():
            if n in trainer._hpfl_classifier_names:
                p.data.copy_(saved_cls[n])

    return per_client


def _compute_fairness(per_client_acc: Dict[str, float]) -> Dict[str, float]:
    accs = list(per_client_acc.values())
    if not accs:
        return {}
    jain = (sum(accs) ** 2) / (len(accs) * sum(a ** 2 for a in accs)) if accs else 0
    sorted_a = sorted(accs)
    n = len(sorted_a)
    cumsum = np.cumsum(sorted_a)
    gini = (2 * sum((i + 1) * v for i, v in enumerate(sorted_a))) / (n * cumsum[-1]) - (n + 1) / n if cumsum[-1] > 0 else 0
    gini = max(0, gini)
    return {
        "mean": round(float(np.mean(accs)), 4),
        "std": round(float(np.std(accs)), 4),
        "min": round(float(min(accs)), 4),
        "max": round(float(max(accs)), 4),
        "jain_index": round(float(jain), 4),
        "gini": round(float(gini), 4),
    }


# ======================================================================
# Article 71 Opt-Out: remove fraction of training images per client
# ======================================================================

def apply_optout_imaging(trainer, optout_rate, seed):
    if optout_rate <= 0:
        orig = {cid: len(y) for cid, (X, y) in trainer.client_data.items()}
        return {"original": orig, "after": dict(orig), "removed": 0, "total": sum(orig.values())}

    orig_counts = {}
    post_counts = {}
    total_removed = 0

    for cid, (X, y) in trainer.client_data.items():
        rng = np.random.RandomState(seed * 1000 + cid)
        n = len(y)
        orig_counts[cid] = n
        n_keep = max(1, int(n * (1 - optout_rate)))
        indices = rng.choice(n, size=n_keep, replace=False)
        indices.sort()
        trainer.client_data[cid] = (X[indices], y[indices])
        post_counts[cid] = n_keep
        total_removed += n - n_keep

    total_orig = sum(orig_counts.values())
    return {
        "original": orig_counts,
        "after": post_counts,
        "removed": total_removed,
        "total": total_orig,
        "actual_rate": round(total_removed / total_orig, 4) if total_orig > 0 else 0,
    }


# ======================================================================
# Training
# ======================================================================

def run_single_dp_optout(
    ds_name: str, data_dir: str, algorithm: str,
    optout_rate: float, seed: int,
    config: Dict, es_config: Dict,
    exp_idx: int, total_exps: int,
    checkpoint_data: Dict = None, exp_key: str = None,
    trainer_ckpt_path: str = None,
) -> Dict[str, Any]:
    start = time.time()
    num_rounds = config["num_rounds"]

    cfg = {**config}
    if ds_name in DATASET_OVERRIDES:
        cfg.update(DATASET_OVERRIDES[ds_name])

    trainer = ImageFederatedTrainer(
        data_dir=data_dir,
        num_clients=cfg["num_clients"],
        algorithm=algorithm,
        local_epochs=cfg["local_epochs"],
        batch_size=cfg["batch_size"],
        learning_rate=cfg["learning_rate"],
        is_iid=cfg["is_iid"],
        alpha=cfg["alpha"],
        mu=cfg.get("mu", 0.1),
        seed=seed,
        model_type=cfg["model_type"],
        freeze_backbone=cfg.get("freeze_backbone", False),
        freeze_level=cfg.get("freeze_level"),
        use_fedbn=cfg.get("use_fedbn", False),
        use_class_weights=cfg.get("use_class_weights", True),
        use_amp=cfg.get("use_amp", True),
        # DP parameters
        dp_enabled=True,
        dp_epsilon=DP_EPSILON,
        dp_clip_norm=DP_CLIP_NORM,
    )
    trainer.num_rounds = num_rounds

    # Apply opt-out BEFORE training
    optout_info = apply_optout_imaging(trainer, optout_rate, seed)
    log(f"  DP(eps={DP_EPSILON}) + Opt-out {optout_rate:.0%}: removed {optout_info['removed']}/{optout_info['total']} "
        f"samples (actual: {optout_info.get('actual_rate', 0):.1%})")

    # Recalculate class weights after opt-out
    if cfg.get("use_class_weights", True) and optout_rate > 0:
        all_labels = np.concatenate([y for _, y in trainer.client_data.values()])
        counts = np.bincount(all_labels, minlength=trainer.num_classes)
        if counts.min() > 0 and counts.max() / counts.min() > 1.5:
            weights = 1.0 / counts.astype(float)
            weights = weights / weights.sum() * len(weights)
            trainer.class_weights = torch.FloatTensor(weights).to(trainer.device)
            trainer.criterion = torch.nn.CrossEntropyLoss(weight=trainer.class_weights)

    es = EarlyStoppingMonitor(
        **{k: v for k, v in es_config.items() if k != "enabled"}
    ) if es_config.get("enabled") else None

    history = []
    best_acc = 0.0
    best_round = 0
    start_round = 0

    # Resume mid-experiment if trainer checkpoint exists
    in_prog = checkpoint_data.get("in_progress") if checkpoint_data else None
    if (in_prog and in_prog.get("key") == exp_key
            and trainer_ckpt_path and Path(trainer_ckpt_path).exists()):
        try:
            start_round = trainer.load_checkpoint(trainer_ckpt_path)
            history = in_prog.get("history", [])
            best_acc = in_prog.get("best_acc", 0.0)
            best_round = in_prog.get("best_round", 0)
            if es and history:
                for h in history:
                    es.check(h["round"], {"accuracy": h["accuracy"]})
            log(f"  RESUMED from round {start_round} (best={best_acc:.1%})")
        except Exception as e:
            log(f"  WARNING: could not resume trainer ({e}), restarting from R1")
            start_round = 0
            history = []
            best_acc = 0.0
            best_round = 0

    opt_label = f"opt={int(optout_rate*100)}%+DP"

    for r in range(start_round, num_rounds):
        rr = trainer.train_round(r)

        # Per-client training metrics with per-epoch breakdown
        client_metrics = [
            {
                "client_id": cr.client_id,
                "train_loss": round(cr.train_loss, 6),
                "train_acc": round(cr.train_acc, 6),
                "num_samples": cr.num_samples,
                "epochs_completed": cr.epochs_completed,
                "epoch_metrics": cr.epoch_metrics or [],
            }
            for cr in rr.client_results
        ]

        metrics = {
            "round": r + 1,
            "accuracy": rr.global_acc,
            "loss": rr.global_loss,
            "f1": rr.global_f1,
            "precision": rr.global_precision,
            "recall": rr.global_recall,
            "auc": rr.global_auc,
            "time_seconds": round(rr.time_seconds, 2),
            "client_results": client_metrics,
        }
        history.append(metrics)

        if rr.global_acc > best_acc:
            best_acc = rr.global_acc
            best_round = r + 1

        log(f"[{exp_idx}/{total_exps}] {ds_name} | {algorithm} | {opt_label} | s{seed} | "
            f"R{r+1}/{num_rounds} | Acc:{rr.global_acc:.1%} | Best:{best_acc:.1%}(r{best_round})")

        # Save after EVERY round
        if checkpoint_data is not None and exp_key:
            if trainer_ckpt_path:
                try:
                    trainer.save_checkpoint(trainer_ckpt_path)
                except Exception:
                    pass
            checkpoint_data["in_progress"] = {
                "key": exp_key,
                "dataset": ds_name,
                "algorithm": algorithm,
                "seed": seed,
                "optout_rate": optout_rate,
                "dp_epsilon": DP_EPSILON,
                "round": r + 1,
                "total_rounds": num_rounds,
                "best_acc": best_acc,
                "best_round": best_round,
                "history": history,
                "elapsed_seconds": round(time.time() - start, 1),
            }
            save_checkpoint(checkpoint_data)

        if es and es.check(r + 1, {"accuracy": rr.global_acc}):
            log(f"  -> Early stop at R{r+1} (best={best_acc:.1%} at r{best_round})")
            break

    per_client_acc = _evaluate_per_client(trainer)
    fairness = _compute_fairness(per_client_acc)
    elapsed = time.time() - start

    result = {
        "dataset": ds_name,
        "algorithm": algorithm,
        "seed": seed,
        "optout_rate": optout_rate,
        "dp_enabled": True,
        "dp_epsilon": DP_EPSILON,
        "dp_clip_norm": DP_CLIP_NORM,
        "optout_info": {
            "samples_original": optout_info["total"],
            "samples_after": sum(optout_info["after"].values()),
            "actual_rate": optout_info.get("actual_rate", 0),
            "per_client_original": {str(k): v for k, v in optout_info["original"].items()},
            "per_client_after": {str(k): v for k, v in optout_info["after"].items()},
        },
        "history": history,
        "final_metrics": history[-1] if history else {},
        "per_client_acc": per_client_acc,
        "fairness": fairness,
        "runtime_seconds": round(elapsed, 1),
        "config": cfg,
        "stopped_early": es is not None and es.counter >= es.patience,
        "actual_rounds": len(history),
        "best_metrics": {"accuracy": best_acc, "round": best_round},
        "best_round": best_round,
    }

    if checkpoint_data is not None:
        checkpoint_data.pop("in_progress", None)
    if trainer_ckpt_path:
        try:
            Path(trainer_ckpt_path).unlink(missing_ok=True)
        except OSError:
            pass

    del trainer
    _cleanup_gpu()
    return result


# ======================================================================
# Dataset download (for RunPod / cloud environments)
# ======================================================================

def download_datasets():
    try:
        import kagglehub
    except ImportError:
        print("Installing kagglehub...")
        os.system(f"{sys.executable} -m pip install -q kagglehub")
        import kagglehub

    os.environ['KAGGLE_API_TOKEN'] = 'KGAT_edd561c1bc682c9ad06930bacd164431'

    data_dir = FRAMEWORK_DIR / "data"
    data_dir.mkdir(exist_ok=True)

    # --- Chest X-Ray ---
    cx_dir = data_dir / "chest_xray"
    if not cx_dir.exists() or not any(cx_dir.iterdir()):
        log("Downloading Chest X-Ray (~2.3 GB)...")
        cache = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")
        cx_dir.mkdir(exist_ok=True)
        for item in ['train', 'test', 'val']:
            src = Path(cache) / "chest_xray" / item
            if not src.exists():
                src = Path(cache) / item
            dst = cx_dir / item
            if src.exists() and not dst.exists():
                shutil.copytree(str(src), str(dst))
        macos_dir = cx_dir / "__MACOSX"
        if macos_dir.exists():
            shutil.rmtree(str(macos_dir))
        log(f"  Chest X-Ray ready: {cx_dir}")
    else:
        log(f"  Chest X-Ray already present: {cx_dir}")

    # --- Skin Cancer ---
    sc_dir = data_dir / "Skin Cancer"
    if not sc_dir.exists() or not any(sc_dir.iterdir()):
        log("Downloading Skin Cancer (~325 MB)...")
        cache = kagglehub.dataset_download("fanconic/skin-cancer-malignant-vs-benign")
        shutil.copytree(cache, str(sc_dir))
        log(f"  Skin Cancer ready: {sc_dir}")
    else:
        log(f"  Skin Cancer already present: {sc_dir}")

    # --- Brain Tumor ---
    bt_dir = data_dir / "Brain_Tumor"
    if not bt_dir.exists() or not any(bt_dir.iterdir()):
        log("Downloading Brain Tumor (~250 MB)...")
        cache = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
        bt_dir.mkdir(exist_ok=True)
        for root, dirs, files in os.walk(cache):
            for d in dirs:
                d_lower = d.lower()
                if d_lower in ['glioma', 'meningioma', 'pituitary', 'notumor', 'no_tumor', 'healthy']:
                    target = 'healthy' if d_lower in ['notumor', 'no_tumor'] else d_lower
                    src = Path(root) / d
                    dst_dir = bt_dir / target
                    if not dst_dir.exists():
                        shutil.copytree(str(src), str(dst_dir))
                    else:
                        for f in os.listdir(str(src)):
                            sf = src / f
                            df = dst_dir / f
                            if sf.is_file() and not df.exists():
                                shutil.copy2(str(sf), str(df))
        log(f"  Brain Tumor ready: {bt_dir}")
    else:
        log(f"  Brain Tumor already present: {bt_dir}")

    # Clean macOS ._* files
    removed = 0
    for root, dirs, files in os.walk(str(data_dir)):
        for f in files:
            if f.startswith("._"):
                try:
                    os.unlink(os.path.join(root, f))
                    removed += 1
                except OSError:
                    pass
    if removed:
        log(f"  Cleaned {removed} macOS ._* files")


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="FL-EHDS Imaging — DP + Opt-Out Combined (EHDS Realistic Scenario)"
    )
    parser.add_argument("--fresh", action="store_true", help="Discard checkpoint, start fresh")
    parser.add_argument("--quick", action="store_true", help="Quick validation (3 rounds, 1 seed)")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Run only one dataset: chest_xray, Brain_Tumor, Skin_Cancer")
    parser.add_argument("--download", action="store_true",
                        help="Download datasets before running (for cloud/RunPod)")
    args = parser.parse_args()

    global _log_file
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    _log_file = open(OUTPUT_DIR / LOG_FILE, "a")

    if args.download:
        download_datasets()

    config = {**IMAGING_CONFIG}
    es_config = {**EARLY_STOPPING}

    ds_filter = None
    if args.dataset:
        if args.dataset not in IMAGING_DATASETS:
            print(f"ERROR: unknown dataset '{args.dataset}'. Choose: {list(IMAGING_DATASETS.keys())}")
            sys.exit(1)
        ds_filter = [args.dataset]

    if args.quick:
        config["num_rounds"] = 3
        config["local_epochs"] = 1
        es_config["enabled"] = False

    experiments = build_experiments(datasets=ds_filter)
    total_exps = len(experiments)

    checkpoint_data = None
    if not args.fresh:
        checkpoint_data = load_checkpoint()
        if checkpoint_data:
            done = len(checkpoint_data.get("completed", {}))
            log(f"AUTO-RESUMED: {done}/{total_exps} completed")
    elif args.fresh:
        ckpt_path = OUTPUT_DIR / CHECKPOINT_FILE
        if ckpt_path.exists():
            ckpt_path.unlink()
        log("FRESH start — previous checkpoint deleted")

    if checkpoint_data is None:
        checkpoint_data = {
            "completed": {},
            "metadata": {
                "total_experiments": total_exps,
                "purpose": "Imaging DP+Opt-out combined: realistic EHDS scenario (eps=5 + GDPR erasure)",
                "algorithms": ALGORITHMS,
                "optout_rates": OPTOUT_RATES,
                "dp_epsilon": DP_EPSILON,
                "dp_clip_norm": DP_CLIP_NORM,
                "datasets": ds_filter or list(IMAGING_DATASETS.keys()),
                "seeds": SEEDS,
                "start_time": datetime.now().isoformat(),
                "last_save": None,
                "version": "imaging_dp_optout_v1",
            }
        }

    # Graceful shutdown
    _interrupted = [False]

    def _signal_handler(signum, frame):
        if _interrupted[0]:
            sys.exit(1)
        _interrupted[0] = True
        done = len(checkpoint_data.get("completed", {}))
        save_checkpoint(checkpoint_data)
        ds_arg = f" --dataset {args.dataset}" if args.dataset else ""
        print(f"\n  Checkpoint salvato: {done}/{total_exps}")
        print(f"  Per riprendere: python -m benchmarks.run_imaging_dp_optout{ds_arg}")
        sys.exit(0)

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    mode = "QUICK" if args.quick else "FULL"
    ds_label = args.dataset if args.dataset else "ALL (3 datasets)"
    log(f"\n{'='*70}")
    log(f"  FL-EHDS Imaging — DP (eps={DP_EPSILON}) + Article 71 Opt-Out ({mode})")
    log(f"  {total_exps} experiments")
    log(f"  Algorithms: {ALGORITHMS}")
    log(f"  Opt-out rates: {[f'{r:.0%}' for r in OPTOUT_RATES]}")
    log(f"  DP: epsilon={DP_EPSILON}, clip_norm={DP_CLIP_NORM}")
    log(f"  Datasets: {ds_label}")
    log(f"  Seeds: {SEEDS}")
    log(f"  Device: {_detect_device(None)}")
    log(f"  Output: {OUTPUT_DIR / CHECKPOINT_FILE}")
    log(f"{'='*70}")

    trainer_ckpt_path = str(OUTPUT_DIR / TRAINER_STATE_FILE)

    global_start = time.time()
    completed_count = len(checkpoint_data.get("completed", {}))

    for exp_idx, (ds_name, algo, rate, seed) in enumerate(experiments, 1):
        key = experiment_key(ds_name, algo, rate, seed)

        if key in checkpoint_data.get("completed", {}):
            continue
        if _interrupted[0]:
            break

        opt_label = f"opt={int(rate*100)}%+DP(eps={int(DP_EPSILON)})"
        log(f"\n--- [{exp_idx}/{total_exps}] {ds_name} | {algo} | {opt_label} | seed={seed} ---")

        try:
            data_dir = IMAGING_DATASETS[ds_name]["data_dir"]
            result = run_single_dp_optout(
                ds_name=ds_name,
                data_dir=data_dir,
                algorithm=algo,
                optout_rate=rate,
                seed=seed,
                config=config,
                es_config=es_config,
                exp_idx=exp_idx,
                total_exps=total_exps,
                checkpoint_data=checkpoint_data,
                exp_key=key,
                trainer_ckpt_path=trainer_ckpt_path,
            )

            checkpoint_data["completed"][key] = result
            completed_count += 1
            save_checkpoint(checkpoint_data)

            best_acc = result.get("best_metrics", {}).get("accuracy", 0)
            es_info = f" ES@R{result['actual_rounds']}" if result.get("stopped_early") else ""
            log(f"--- Done: {ds_name} | {algo} | {opt_label} | s{seed} | "
                f"Best={best_acc:.1%}{es_info} | {result['runtime_seconds']:.0f}s | "
                f"[{completed_count}/{total_exps}] ---")

            elapsed = time.time() - global_start
            if completed_count > 0:
                avg = elapsed / completed_count
                remaining = (total_exps - completed_count) * avg
                eta = str(timedelta(seconds=int(remaining)))
                log(f"  ETA: ~{eta}")

        except Exception as e:
            log(f"ERROR in {key}: {e}")
            traceback.print_exc()
            checkpoint_data["completed"][key] = {
                "dataset": ds_name, "algorithm": algo, "seed": seed,
                "optout_rate": rate, "dp_epsilon": DP_EPSILON,
                "error": str(e), "traceback": traceback.format_exc(),
            }
            save_checkpoint(checkpoint_data)

    elapsed_total = time.time() - global_start

    # Summary table
    log(f"\n{'='*70}")
    log(f"  COMPLETED: {completed_count}/{total_exps}")
    log(f"  Total time: {str(timedelta(seconds=int(elapsed_total)))}")
    log(f"{'='*70}")

    log(f"\n  DP+Opt-Out Impact Summary (Best Accuracy, mean over seeds):")
    header = f"  {'Dataset':<14} {'Algorithm':<10}" + "".join(f" {'opt='+str(int(r*100))+'%':>9}" for r in OPTOUT_RATES)
    log(header)
    log(f"  {'-'*len(header)}")

    completed = checkpoint_data.get("completed", {})
    ds_list = ds_filter or list(IMAGING_DATASETS.keys())
    for ds_name in ds_list:
        for algo in ALGORITHMS:
            row = f"  {ds_name:<14} {algo:<10}"
            for rate in OPTOUT_RATES:
                accs = []
                for seed in SEEDS:
                    k = experiment_key(ds_name, algo, rate, seed)
                    r = completed.get(k, {})
                    if "error" not in r and r:
                        accs.append(r.get("best_metrics", {}).get("accuracy", 0))
                if accs:
                    row += f" {100*np.mean(accs):>8.1f}%"
                else:
                    row += f" {'--':>9}"
            log(row)

    n_err = sum(1 for v in completed.values() if "error" in v)
    if n_err:
        log(f"\n  ERRORS ({n_err}):")
        for k, v in completed.items():
            if "error" in v:
                log(f"    {k}: {v['error']}")

    checkpoint_data["metadata"]["end_time"] = datetime.now().isoformat()
    checkpoint_data["metadata"]["total_elapsed"] = elapsed_total
    save_checkpoint(checkpoint_data)
    if _log_file:
        _log_file.close()


if __name__ == "__main__":
    main()
