"""
Centralized baseline trainers and FL vs centralized comparison utilities.
"""

import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass
from copy import deepcopy
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from terminal.training.data_generation import _detect_device, generate_healthcare_data
from terminal.training.models import HealthcareMLP, HealthcareCNN, HealthcareResNet, load_image_dataset
from terminal.training.federated import ClientResult, RoundResult, FederatedTrainer
from terminal.training.federated_image import ImageFederatedTrainer


# =============================================================================
# CENTRALIZED BASELINE TRAINER
# =============================================================================

@dataclass
class CentralizedResult:
    """Result from one centralized training epoch."""
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_f1: float
    val_precision: float
    val_recall: float
    val_auc: float
    time_seconds: float


class CentralizedTrainer:
    """
    Centralized baseline trainer - simulates scenario where all hospital data
    is pooled into a single central server (no federated learning).

    This serves as an upper bound for FL performance comparison.
    """

    def __init__(
        self,
        num_clients: int = 5,
        samples_per_client: int = 200,
        batch_size: int = 32,
        learning_rate: float = 0.01,
        is_iid: bool = False,
        alpha: float = 0.5,
        seed: int = 42,
        device: str = "cpu",
        progress_callback: Optional[Callable] = None,
        val_split: float = 0.2,  # Validation split ratio
    ):
        self.num_clients = num_clients
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.seed = seed
        self.device = torch.device(device)
        self.progress_callback = progress_callback
        self.val_split = val_split

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate data using same function as FL
        client_train_data, client_test_data = generate_healthcare_data(
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            is_iid=is_iid,
            alpha=alpha,
            seed=seed
        )

        # Combine all client data into one dataset
        all_X = []
        all_y = []
        for client_id in range(num_clients):
            X, y = client_train_data[client_id]
            all_X.append(X)
            all_y.append(y)

        self.X_all = np.vstack(all_X)
        self.y_all = np.concatenate(all_y)

        # Split into train/val
        n_samples = len(self.y_all)
        n_val = int(n_samples * val_split)
        indices = np.random.permutation(n_samples)
        val_indices = indices[:n_val]
        train_indices = indices[n_val:]

        self.X_train = self.X_all[train_indices]
        self.y_train = self.y_all[train_indices]
        self.X_val = self.X_all[val_indices]
        self.y_val = self.y_all[val_indices]

        # Initialize model
        self.model = HealthcareMLP().to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.CrossEntropyLoss()

        # History
        self.history = []

    def _get_dataloader(self, X: np.ndarray, y: np.ndarray, shuffle: bool = True) -> DataLoader:
        """Create DataLoader from numpy arrays."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle)

    def _evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model on given data with comprehensive metrics."""
        self.model.eval()

        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, y_tensor)
            preds = outputs.argmax(dim=1)
            probs = torch.softmax(outputs, dim=1)[:, 1]

        preds_np = preds.cpu().numpy()
        labels_np = y_tensor.cpu().numpy()
        probs_np = probs.cpu().numpy()

        # Calculate metrics
        accuracy = (preds_np == labels_np).mean()

        tp = ((preds_np == 1) & (labels_np == 1)).sum()
        fp = ((preds_np == 1) & (labels_np == 0)).sum()
        fn = ((preds_np == 0) & (labels_np == 1)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        # AUC-ROC
        sorted_indices = np.argsort(probs_np)[::-1]
        sorted_labels = labels_np[sorted_indices]
        n_pos = (labels_np == 1).sum()
        n_neg = (labels_np == 0).sum()

        if n_pos > 0 and n_neg > 0:
            tpr_sum = 0.0
            tp_count = 0
            for label in sorted_labels:
                if label == 1:
                    tp_count += 1
                else:
                    tpr_sum += tp_count / n_pos
            auc = tpr_sum / n_neg
        else:
            auc = 0.5

        return {
            "loss": loss.item(),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "auc": float(auc),
        }

    def train_epoch(self, epoch: int) -> CentralizedResult:
        """Train for one epoch."""
        start_time = time.time()

        if self.progress_callback:
            self.progress_callback("epoch_start", epoch=epoch + 1)

        self.model.train()
        train_loader = self._get_dataloader(self.X_train, self.y_train)

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for batch_X, batch_y in train_loader:
            self.optimizer.zero_grad()

            outputs = self.model(batch_X)
            loss = self.criterion(outputs, batch_y)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(batch_y)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == batch_y).sum().item()
            total_samples += len(batch_y)

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # Evaluate on validation set
        val_metrics = self._evaluate(self.X_val, self.y_val)

        elapsed = time.time() - start_time

        result = CentralizedResult(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["accuracy"],
            val_f1=val_metrics["f1"],
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
            val_auc=val_metrics["auc"],
            time_seconds=elapsed
        )

        self.history.append(result)

        if self.progress_callback:
            self.progress_callback(
                "epoch_end",
                epoch=epoch + 1,
                train_loss=train_loss,
                train_acc=train_acc,
                val_loss=val_metrics["loss"],
                val_acc=val_metrics["accuracy"],
                val_f1=val_metrics["f1"],
                val_auc=val_metrics["auc"],
                time=elapsed
            )

        return result

    def evaluate_on_all_data(self) -> Dict[str, float]:
        """Evaluate model on ALL data (train + val) for fair comparison with FL."""
        return self._evaluate(self.X_all, self.y_all)

    def get_data_stats(self) -> Dict:
        """Get statistics about the centralized dataset."""
        unique, counts = np.unique(self.y_all, return_counts=True)
        return {
            "total_samples": len(self.y_all),
            "train_samples": len(self.y_train),
            "val_samples": len(self.y_val),
            "label_distribution": dict(zip(unique.tolist(), counts.tolist())),
            "class_balance": counts.min() / counts.max() if len(counts) > 1 else 1.0
        }


# =============================================================================
# CENTRALIZED IMAGE TRAINER (for imaging datasets)
# =============================================================================


class CentralizedImageTrainer:
    """
    Centralized baseline trainer for medical image classification.
    Pools all hospital image data into a single central model.
    Serves as the upper bound for FL performance comparison.
    """

    def __init__(
        self,
        data_dir: str,
        num_clients: int = 5,
        batch_size: int = 32,
        learning_rate: float = 0.001,
        is_iid: bool = False,
        alpha: float = 0.5,
        seed: int = 42,
        device: str = None,
        img_size: int = 128,
        progress_callback: Optional[Callable] = None,
        model_type: str = "resnet18",
        freeze_backbone: bool = False,
        use_class_weights: bool = True,
    ):
        self.batch_size = batch_size
        self.seed = seed
        self.progress_callback = progress_callback
        self.model_type = model_type

        # Auto-adjust for ResNet
        if model_type == "resnet18" and img_size < 224:
            img_size = 224
        if model_type == "resnet18" and learning_rate == 0.001:
            learning_rate = 0.0005
        self.learning_rate = learning_rate
        self.img_size = img_size

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.device = _detect_device(device)

        # Load data using same function as ImageFederatedTrainer
        client_train, client_test, class_names, num_classes = load_image_dataset(
            data_dir=data_dir,
            num_clients=num_clients,
            is_iid=is_iid,
            alpha=alpha,
            img_size=img_size,
            seed=seed,
            test_split=0.2,
        )

        self.num_classes = num_classes
        self.class_names = class_names
        self.num_clients = num_clients
        self.client_train_data = client_train
        self.client_test_data = client_test

        # Pool ALL client data into single centralized dataset
        all_X_train = [client_train[c][0] for c in sorted(client_train.keys())]
        all_y_train = [client_train[c][1] for c in sorted(client_train.keys())]
        all_X_test = [client_test[c][0] for c in sorted(client_test.keys())]
        all_y_test = [client_test[c][1] for c in sorted(client_test.keys())]

        self.X_train = np.concatenate(all_X_train, axis=0)
        self.y_train = np.concatenate(all_y_train, axis=0)
        self.X_test = np.concatenate(all_X_test, axis=0)
        self.y_test = np.concatenate(all_y_test, axis=0)

        print(f"  Centralized: {len(self.y_train)} train + {len(self.y_test)} test samples pooled")

        # Class-weighted loss for imbalanced datasets
        class_weights = None
        if use_class_weights:
            counts = np.bincount(self.y_train, minlength=num_classes)
            if counts.min() > 0 and counts.max() / counts.min() > 1.5:
                weights = len(self.y_train) / (num_classes * counts + 1e-8)
                weights = weights / weights.mean()
                class_weights = torch.FloatTensor(weights).to(self.device)

        # Same model type as federated
        if model_type == "resnet18":
            self.model = HealthcareResNet(
                num_classes=num_classes,
                pretrained=True,
                freeze_backbone=freeze_backbone,
            ).to(self.device)
        else:
            self.model = HealthcareCNN(num_classes=num_classes).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=1e-5
        )
        self.criterion = nn.CrossEntropyLoss(weight=class_weights)
        self.augmentation = ImageFederatedTrainer._build_augmentation(img_size)
        self.scheduler = None  # Set via set_total_epochs()
        self.history: List[CentralizedResult] = []

    def set_total_epochs(self, total_epochs: int):
        """Enable cosine LR scheduler over total_epochs."""
        from torch.optim.lr_scheduler import CosineAnnealingLR
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=self.learning_rate * 0.1
        )

    def train_epoch(self, epoch: int) -> CentralizedResult:
        """Train for one epoch on pooled data."""
        start_time = time.time()

        if self.progress_callback:
            self.progress_callback("epoch_start", epoch=epoch + 1)

        self.model.train()

        # Shuffle training data
        perm = np.random.permutation(len(self.y_train))
        X_shuffled = self.X_train[perm]
        y_shuffled = self.y_train[perm]

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for i in range(0, len(y_shuffled), self.batch_size):
            X_batch = torch.FloatTensor(X_shuffled[i:i+self.batch_size]).to(self.device)
            y_batch = torch.LongTensor(y_shuffled[i:i+self.batch_size]).to(self.device)

            # Same augmentation as federated training
            if self.augmentation is not None:
                augmented = []
                for img in X_batch:
                    augmented.append(self.augmentation(img))
                X_batch = torch.stack(augmented)

            self.optimizer.zero_grad()
            outputs = self.model(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(y_batch)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)

        train_loss = total_loss / max(total_samples, 1)
        train_acc = total_correct / max(total_samples, 1)

        # Step LR scheduler after each epoch
        if self.scheduler is not None:
            self.scheduler.step()

        # Evaluate on held-out test set
        val_metrics = self._evaluate()
        elapsed = time.time() - start_time

        result = CentralizedResult(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["accuracy"],
            val_f1=val_metrics["f1"],
            val_precision=val_metrics["precision"],
            val_recall=val_metrics["recall"],
            val_auc=val_metrics["auc"],
            time_seconds=elapsed,
        )
        self.history.append(result)

        if self.progress_callback:
            self.progress_callback(
                "epoch_end", epoch=epoch + 1,
                train_loss=train_loss, train_acc=train_acc,
                val_acc=val_metrics["accuracy"], val_f1=val_metrics["f1"],
                val_auc=val_metrics["auc"], time=elapsed,
            )

        return result

    def _evaluate(self) -> Dict[str, float]:
        """Evaluate model on held-out test data."""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []

        with torch.no_grad():
            for i in range(0, len(self.y_test), self.batch_size):
                X_batch = torch.FloatTensor(self.X_test[i:i+self.batch_size]).to(self.device)
                y_batch = torch.LongTensor(self.y_test[i:i+self.batch_size]).to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * len(y_batch)

                preds = outputs.argmax(dim=1)
                probs = torch.softmax(outputs, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_batch.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = (all_preds == all_labels).mean()
        unique_classes = np.unique(all_labels)
        precisions, recalls, f1s = [], [], []
        for cls in unique_classes:
            tp = ((all_preds == cls) & (all_labels == cls)).sum()
            fp = ((all_preds == cls) & (all_labels != cls)).sum()
            fn = ((all_preds != cls) & (all_labels == cls)).sum()
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            precisions.append(p)
            recalls.append(r)
            f1s.append(f)

        auc = 0.5
        try:
            from sklearn.metrics import roc_auc_score
            if len(unique_classes) == 2:
                auc = roc_auc_score(all_labels, all_probs[:, 1])
            elif len(unique_classes) > 2 and all_probs.shape[1] >= len(unique_classes):
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
        except Exception:
            auc = float(accuracy)

        return {
            "loss": total_loss / max(len(all_labels), 1),
            "accuracy": float(accuracy),
            "precision": float(np.mean(precisions)),
            "recall": float(np.mean(recalls)),
            "f1": float(np.mean(f1s)),
            "auc": float(auc),
        }

    def get_predictions(self) -> tuple:
        """Run inference on test set. Returns (preds, probs, labels)."""
        self.model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for i in range(0, len(self.y_test), self.batch_size):
                X_batch = torch.FloatTensor(self.X_test[i:i+self.batch_size]).to(self.device)
                y_batch = self.y_test[i:i+self.batch_size]
                outputs = self.model(X_batch)
                all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1).cpu().numpy())
                all_labels.extend(y_batch)
        return np.array(all_preds), np.array(all_probs), np.array(all_labels)

    def get_model(self) -> nn.Module:
        """Return the trained model."""
        return self.model

    def get_data_stats(self) -> Dict:
        """Get per-client and overall data statistics."""
        per_client = {}
        for cid in sorted(self.client_train_data.keys()):
            _, y_tr = self.client_train_data[cid]
            _, y_te = self.client_test_data[cid]
            unique, counts = np.unique(y_tr, return_counts=True)
            per_client[cid] = {
                "train_samples": len(y_tr),
                "test_samples": len(y_te),
                "class_distribution": dict(zip(unique.tolist(), counts.tolist())),
            }

        unique_all, counts_all = np.unique(self.y_train, return_counts=True)
        return {
            "total_train": len(self.y_train),
            "total_test": len(self.y_test),
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "overall_distribution": dict(zip(unique_all.tolist(), counts_all.tolist())),
            "per_client": per_client,
        }


# =============================================================================
# FL vs CENTRALIZED COMPARISON
# =============================================================================

def run_fl_vs_centralized_comparison(
    num_clients: int = 5,
    samples_per_client: int = 200,
    fl_rounds: int = 30,
    local_epochs: int = 3,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    is_iid: bool = False,
    alpha: float = 0.5,
    seeds: List[int] = [42, 123, 456],
    algorithms: List[str] = ["FedAvg"],
    verbose: bool = True,
) -> Dict:
    """
    Run comparison between FL algorithms and centralized baseline.

    Runs multiple seeds and returns mean ± std for all metrics.
    For fair comparison, centralized training uses same total epochs:
        centralized_epochs = fl_rounds * local_epochs

    Returns dict with results for each algorithm plus 'Centralized' baseline.
    """
    results = {}

    # Total epochs for centralized (to match FL computation)
    centralized_epochs = fl_rounds * local_epochs

    for algo in algorithms + ["Centralized"]:
        results[algo] = {
            "accuracy": [],
            "loss": [],
            "f1": [],
            "precision": [],
            "recall": [],
            "auc": [],
            "time": [],
        }

    for seed_idx, seed in enumerate(seeds):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Seed {seed_idx + 1}/{len(seeds)}: {seed}")
            print('='*60)

        # Run centralized baseline
        if verbose:
            print(f"\n[Centralized] Training for {centralized_epochs} epochs...")

        centralized = CentralizedTrainer(
            num_clients=num_clients,
            samples_per_client=samples_per_client,
            batch_size=batch_size,
            learning_rate=learning_rate,
            is_iid=is_iid,
            alpha=alpha,
            seed=seed,
        )

        start_time = time.time()
        for epoch in range(centralized_epochs):
            centralized.train_epoch(epoch)

        central_time = time.time() - start_time

        # Evaluate on all data for fair comparison
        central_metrics = centralized.evaluate_on_all_data()

        results["Centralized"]["accuracy"].append(central_metrics["accuracy"])
        results["Centralized"]["loss"].append(central_metrics["loss"])
        results["Centralized"]["f1"].append(central_metrics["f1"])
        results["Centralized"]["precision"].append(central_metrics["precision"])
        results["Centralized"]["recall"].append(central_metrics["recall"])
        results["Centralized"]["auc"].append(central_metrics["auc"])
        results["Centralized"]["time"].append(central_time)

        if verbose:
            print(f"  Accuracy: {central_metrics['accuracy']:.4f}, "
                  f"F1: {central_metrics['f1']:.4f}, "
                  f"AUC: {central_metrics['auc']:.4f}")

        # Run FL algorithms
        for algo in algorithms:
            if verbose:
                print(f"\n[{algo}] Training for {fl_rounds} rounds...")

            trainer = FederatedTrainer(
                num_clients=num_clients,
                samples_per_client=samples_per_client,
                algorithm=algo,
                local_epochs=local_epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                is_iid=is_iid,
                alpha=alpha,
                seed=seed,
            )

            start_time = time.time()
            for r in range(fl_rounds):
                trainer.train_round(r)

            fl_time = time.time() - start_time

            # Get final metrics
            final = trainer.history[-1]

            results[algo]["accuracy"].append(final.global_acc)
            results[algo]["loss"].append(final.global_loss)
            results[algo]["f1"].append(final.global_f1)
            results[algo]["precision"].append(final.global_precision)
            results[algo]["recall"].append(final.global_recall)
            results[algo]["auc"].append(final.global_auc)
            results[algo]["time"].append(fl_time)

            if verbose:
                print(f"  Accuracy: {final.global_acc:.4f}, "
                      f"F1: {final.global_f1:.4f}, "
                      f"AUC: {final.global_auc:.4f}")

    # Compute mean ± std
    summary = {}
    for algo, metrics in results.items():
        summary[algo] = {}
        for metric_name, values in metrics.items():
            arr = np.array(values)
            summary[algo][metric_name] = {
                "mean": float(arr.mean()),
                "std": float(arr.std()),
                "values": values
            }

    # Compute gap from centralized
    central_acc = summary["Centralized"]["accuracy"]["mean"]
    for algo in algorithms:
        fl_acc = summary[algo]["accuracy"]["mean"]
        gap = central_acc - fl_acc
        gap_pct = (gap / central_acc) * 100 if central_acc > 0 else 0
        summary[algo]["gap_from_centralized"] = {
            "absolute": gap,
            "percentage": gap_pct
        }

    if verbose:
        print("\n" + "="*60)
        print("SUMMARY: FL vs Centralized Comparison")
        print("="*60)
        print(f"\nCentralized Baseline:")
        print(f"  Accuracy: {summary['Centralized']['accuracy']['mean']:.4f} "
              f"± {summary['Centralized']['accuracy']['std']:.4f}")
        print(f"  F1: {summary['Centralized']['f1']['mean']:.4f} "
              f"± {summary['Centralized']['f1']['std']:.4f}")
        print(f"  AUC: {summary['Centralized']['auc']['mean']:.4f} "
              f"± {summary['Centralized']['auc']['std']:.4f}")

        for algo in algorithms:
            print(f"\n{algo}:")
            print(f"  Accuracy: {summary[algo]['accuracy']['mean']:.4f} "
                  f"± {summary[algo]['accuracy']['std']:.4f}")
            print(f"  F1: {summary[algo]['f1']['mean']:.4f} "
                  f"± {summary[algo]['f1']['std']:.4f}")
            print(f"  AUC: {summary[algo]['auc']['mean']:.4f} "
                  f"± {summary[algo]['auc']['std']:.4f}")
            print(f"  Gap from Centralized: {summary[algo]['gap_from_centralized']['absolute']:.4f} "
                  f"({summary[algo]['gap_from_centralized']['percentage']:.2f}%)")

    return summary


def generate_comparison_latex_table(summary: Dict) -> str:
    """Generate LaTeX table comparing FL algorithms with centralized baseline."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Comparison of Federated Learning vs Centralized Training}",
        r"\label{tab:fl-vs-centralized}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Algorithm & Accuracy & F1 & AUC & Gap (\%) \\",
        r"\midrule",
    ]

    # Centralized first
    c = summary["Centralized"]
    lines.append(
        f"Centralized (Upper Bound) & "
        f"${c['accuracy']['mean']:.3f} \\pm {c['accuracy']['std']:.3f}$ & "
        f"${c['f1']['mean']:.3f} \\pm {c['f1']['std']:.3f}$ & "
        f"${c['auc']['mean']:.3f} \\pm {c['auc']['std']:.3f}$ & "
        f"--- \\\\"
    )
    lines.append(r"\midrule")

    # FL algorithms
    for algo, metrics in summary.items():
        if algo == "Centralized":
            continue
        gap = metrics.get("gap_from_centralized", {}).get("percentage", 0)
        lines.append(
            f"{algo} & "
            f"${metrics['accuracy']['mean']:.3f} \\pm {metrics['accuracy']['std']:.3f}$ & "
            f"${metrics['f1']['mean']:.3f} \\pm {metrics['f1']['std']:.3f}$ & "
            f"${metrics['auc']['mean']:.3f} \\pm {metrics['auc']['std']:.3f}$ & "
            f"{gap:.1f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    return "\n".join(lines)
