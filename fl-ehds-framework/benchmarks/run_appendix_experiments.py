#!/usr/bin/env python3
"""
FL-EHDS Appendix Experiments
============================
Generates additional data for paper appendix:
- Training time analysis
- Data distribution visualization
- Client participation patterns
- Communication overhead
- Gradient statistics
"""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np

np.random.seed(42)

# ============================================================================
# DATA GENERATION
# ============================================================================

def generate_hospital_data(n_samples: int, bias: float, seed: int):
    """Generate hospital data with specific bias."""
    rng = np.random.RandomState(seed)

    age = rng.normal(55 + bias * 10, 12, n_samples)
    bmi = rng.normal(26 + bias * 3, 4, n_samples)
    bp = rng.normal(130 + bias * 10, 18, n_samples)
    glucose = rng.normal(100 + bias * 20, 25, n_samples)
    chol = rng.normal(200 + bias * 25, 35, n_samples)

    X = np.column_stack([
        (age - 55) / 12, (bmi - 26) / 4, (bp - 130) / 18,
        (glucose - 100) / 25, (chol - 200) / 35
    ])

    risk = 0.3*X[:,0] + 0.2*X[:,1] + 0.25*X[:,2] + 0.15*X[:,3] + 0.1*X[:,4]
    prob = 1 / (1 + np.exp(-risk))
    y = (rng.random(n_samples) < prob).astype(float)

    return X, y, {
        'mean_age': float(np.mean(age)),
        'mean_bmi': float(np.mean(bmi)),
        'mean_bp': float(np.mean(bp)),
        'mean_glucose': float(np.mean(glucose)),
        'mean_chol': float(np.mean(chol)),
        'positive_rate': float(np.mean(y)),
        'n_samples': n_samples
    }

# ============================================================================
# TRAINING WITH DETAILED METRICS
# ============================================================================

class DetailedTrainer:
    def __init__(self):
        self.weights = np.zeros(5)
        self.bias = 0.0

    def get_params(self):
        return {'w': self.weights.copy(), 'b': np.array([self.bias])}

    def set_params(self, p):
        self.weights = p['w'].copy()
        self.bias = float(p['b'][0])

    def predict_proba(self, X):
        return 1 / (1 + np.exp(-np.clip(X @ self.weights + self.bias, -500, 500)))

    def compute_loss(self, X, y):
        p = self.predict_proba(X)
        return -np.mean(y * np.log(p + 1e-7) + (1-y) * np.log(1-p + 1e-7))

    def compute_accuracy(self, X, y):
        return np.mean((self.predict_proba(X) >= 0.5).astype(int) == y)

    def train_step(self, X, y, lr, global_p=None, mu=0.0):
        p = self.predict_proba(X)
        err = p - y
        gw = X.T @ err / len(y)
        gb = np.mean(err)
        if mu > 0 and global_p:
            gw += mu * (self.weights - global_p['w'])
            gb += mu * (self.bias - global_p['b'][0])
        self.weights -= lr * gw
        self.bias -= lr * gb
        return gw, gb

def run_detailed_fl_experiment():
    """Run FL with detailed per-round metrics."""
    print("Running detailed FL experiment...")

    # Setup hospitals with different characteristics
    hospitals_config = [
        {'name': 'Hospital-IT-Roma', 'samples': 400, 'bias': -0.6, 'seed': 100},
        {'name': 'Hospital-DE-Berlin', 'samples': 500, 'bias': -0.2, 'seed': 101},
        {'name': 'Hospital-FR-Paris', 'samples': 350, 'bias': 0.0, 'seed': 102},
        {'name': 'Hospital-ES-Madrid', 'samples': 450, 'bias': 0.3, 'seed': 103},
        {'name': 'Hospital-NL-Amsterdam', 'samples': 380, 'bias': 0.6, 'seed': 104},
    ]

    hospital_data = []
    hospital_stats = []
    for cfg in hospitals_config:
        X, y, stats = generate_hospital_data(cfg['samples'], cfg['bias'], cfg['seed'])
        hospital_data.append((cfg['name'], X, y))
        stats['name'] = cfg['name']
        stats['country'] = cfg['name'].split('-')[1]
        hospital_stats.append(stats)

    X_test, y_test, _ = generate_hospital_data(500, 0.0, 999)

    # Training configuration
    num_rounds = 50
    local_epochs = 3
    batch_size = 32
    lr = 0.1

    global_model = DetailedTrainer()

    # Detailed metrics storage
    results = {
        'hospital_stats': hospital_stats,
        'per_round': [],
        'per_client_per_round': {h[0]: [] for h in hospital_data},
        'gradient_norms': {h[0]: [] for h in hospital_data},
        'training_times': {h[0]: [] for h in hospital_data},
        'communication_bytes': {h[0]: [] for h in hospital_data},
        'participation_matrix': [],
    }

    for r in range(1, num_rounds + 1):
        round_start = time.time()
        global_params = global_model.get_params()

        # Simulate client participation (some clients may drop)
        participation = []
        for i, (name, X, y) in enumerate(hospital_data):
            # 90% participation probability
            participates = np.random.random() > 0.1
            participation.append(1 if participates else 0)

        results['participation_matrix'].append(participation)

        updates = []
        round_metrics = {'round': r, 'clients': []}

        for i, (name, X, y) in enumerate(hospital_data):
            if participation[i] == 0:
                results['per_client_per_round'][name].append({
                    'participated': False, 'accuracy': None, 'loss': None
                })
                results['gradient_norms'][name].append(0)
                results['training_times'][name].append(0)
                results['communication_bytes'][name].append(0)
                continue

            client_start = time.time()

            local = DetailedTrainer()
            local.set_params(global_params)

            n = len(X)
            total_grad_norm = 0
            grad_count = 0

            for _ in range(local_epochs):
                idx = np.random.permutation(n)
                for s in range(0, n, batch_size):
                    e = min(s + batch_size, n)
                    gw, gb = local.train_step(X[idx[s:e]], y[idx[s:e]], lr)
                    total_grad_norm += np.sqrt(np.sum(gw**2) + gb**2)
                    grad_count += 1

            client_time = time.time() - client_start

            local_params = local.get_params()
            update = {k: local_params[k] - global_params[k] for k in global_params}

            # Clip gradients
            norm = np.sqrt(sum(np.sum(v**2) for v in update.values()))
            if norm > 1.0:
                update = {k: v / norm for k, v in update.items()}

            updates.append((update, n))

            # Store metrics
            acc = local.compute_accuracy(X, y)
            loss = local.compute_loss(X, y)

            results['per_client_per_round'][name].append({
                'participated': True,
                'accuracy': float(acc),
                'loss': float(loss),
            })
            results['gradient_norms'][name].append(float(total_grad_norm / grad_count))
            results['training_times'][name].append(float(client_time))

            # Communication: param size * 4 bytes (float32) * 2 (up + down)
            param_bytes = sum(p.size * 4 for p in local_params.values()) * 2
            results['communication_bytes'][name].append(param_bytes)

            round_metrics['clients'].append({
                'name': name,
                'accuracy': float(acc),
                'loss': float(loss),
                'grad_norm': float(total_grad_norm / grad_count),
                'time': float(client_time),
            })

        # Aggregate
        if updates:
            total = sum(n for _, n in updates)
            agg = {}
            for k in updates[0][0]:
                agg[k] = sum(u[k] * n for u, n in updates) / total
            new_params = {k: global_params[k] + agg[k] for k in global_params}
            global_model.set_params(new_params)

        # Global metrics
        test_acc = global_model.compute_accuracy(X_test, y_test)
        test_loss = global_model.compute_loss(X_test, y_test)

        round_metrics['test_accuracy'] = float(test_acc)
        round_metrics['test_loss'] = float(test_loss)
        round_metrics['round_time'] = float(time.time() - round_start)
        round_metrics['participating_clients'] = int(sum(participation))

        results['per_round'].append(round_metrics)

        if r % 10 == 0:
            print(f"  Round {r}: Acc={test_acc:.3f}, Participants={sum(participation)}/5")

    return results

def run_learning_rate_experiment():
    """Experiment with different learning rates."""
    print("\nRunning learning rate experiment...")

    lrs = [0.01, 0.05, 0.1, 0.2, 0.5]
    results = {}

    for lr in lrs:
        print(f"  Testing lr={lr}")

        # Simple setup
        hospitals = []
        for i in range(5):
            bias = 0.5 * (2 * i / 4 - 1)
            X, y, _ = generate_hospital_data(400, bias, 42 + i)
            hospitals.append((X, y))

        X_test, y_test, _ = generate_hospital_data(500, 0.0, 999)

        model = DetailedTrainer()
        accs = []

        for r in range(30):
            global_p = model.get_params()
            updates = []

            for X, y in hospitals:
                local = DetailedTrainer()
                local.set_params(global_p)

                for _ in range(3):
                    idx = np.random.permutation(len(X))
                    for s in range(0, len(X), 32):
                        local.train_step(X[idx[s:s+32]], y[idx[s:s+32]], lr)

                lp = local.get_params()
                updates.append(({k: lp[k] - global_p[k] for k in global_p}, len(X)))

            total = sum(n for _, n in updates)
            agg = {k: sum(u[k]*n for u,n in updates)/total for k in updates[0][0]}
            model.set_params({k: global_p[k] + agg[k] for k in global_p})

            accs.append(float(model.compute_accuracy(X_test, y_test)))

        results[f'lr_{lr}'] = accs

    return results

def run_batch_size_experiment():
    """Experiment with different batch sizes."""
    print("\nRunning batch size experiment...")

    batch_sizes = [8, 16, 32, 64, 128]
    results = {}

    for bs in batch_sizes:
        print(f"  Testing batch_size={bs}")

        hospitals = []
        for i in range(5):
            X, y, _ = generate_hospital_data(400, 0.5 * (2*i/4-1), 42+i)
            hospitals.append((X, y))

        X_test, y_test, _ = generate_hospital_data(500, 0.0, 999)

        model = DetailedTrainer()
        times = []
        accs = []

        for r in range(30):
            start = time.time()
            global_p = model.get_params()
            updates = []

            for X, y in hospitals:
                local = DetailedTrainer()
                local.set_params(global_p)

                for _ in range(3):
                    idx = np.random.permutation(len(X))
                    for s in range(0, len(X), bs):
                        local.train_step(X[idx[s:s+bs]], y[idx[s:s+bs]], 0.1)

                lp = local.get_params()
                updates.append(({k: lp[k] - global_p[k] for k in global_p}, len(X)))

            total = sum(n for _,n in updates)
            agg = {k: sum(u[k]*n for u,n in updates)/total for k in updates[0][0]}
            model.set_params({k: global_p[k] + agg[k] for k in global_p})

            times.append(float(time.time() - start))
            accs.append(float(model.compute_accuracy(X_test, y_test)))

        results[f'bs_{bs}'] = {'times': times, 'accuracies': accs}

    return results

def generate_appendix_figures(detailed_results, lr_results, bs_results, output_dir: Path):
    """Generate all figures for appendix."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure A1: Hospital Data Distribution
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    stats = detailed_results['hospital_stats']
    names = [s['name'].split('-')[1] for s in stats]

    metrics = ['mean_age', 'mean_bmi', 'mean_bp', 'mean_glucose', 'mean_chol', 'positive_rate']
    titles = ['Mean Age', 'Mean BMI', 'Mean Blood Pressure', 'Mean Glucose', 'Mean Cholesterol', 'Positive Rate']
    colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12', '#9b59b6']

    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 3, idx % 3]
        vals = [s[metric] for s in stats]
        bars = ax.bar(names, vals, color=colors, edgecolor='black')
        ax.set_title(title, fontsize=11)
        ax.set_ylabel(metric.replace('_', ' ').title())
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, v + 0.01*max(vals),
                    f'{v:.1f}' if metric != 'positive_rate' else f'{v:.1%}',
                    ha='center', fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'figA1_data_distribution.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figA1_data_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure A2: Per-Client Training Time
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, times) in enumerate(detailed_results['training_times'].items()):
        short_name = name.split('-')[1]
        ax.plot(range(1, len(times)+1), times, label=short_name, lw=1.5)

    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Training Time (s)', fontsize=11)
    ax.set_title('Per-Hospital Training Time per Round', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'figA2_training_times.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figA2_training_times.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure A3: Client Participation Matrix
    fig, ax = plt.subplots(figsize=(12, 4))

    participation = np.array(detailed_results['participation_matrix']).T
    im = ax.imshow(participation, aspect='auto', cmap='Blues', interpolation='nearest')

    ax.set_yticks(range(5))
    ax.set_yticklabels([s['name'].split('-')[1] for s in stats])
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Hospital', fontsize=11)
    ax.set_title('Client Participation Matrix (Blue=Participated)', fontsize=12)

    # Add participation rate
    for i in range(5):
        rate = np.mean(participation[i])
        ax.text(51, i, f'{rate:.0%}', va='center', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / 'figA3_participation_matrix.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figA3_participation_matrix.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure A4: Gradient Norms Evolution
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, norms in detailed_results['gradient_norms'].items():
        short_name = name.split('-')[1]
        # Filter out zeros (non-participation)
        valid_norms = [n for n in norms if n > 0]
        valid_rounds = [i+1 for i, n in enumerate(norms) if n > 0]
        ax.plot(valid_rounds, valid_norms, 'o-', label=short_name, markersize=3, lw=1)

    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Average Gradient Norm', fontsize=11)
    ax.set_title('Gradient Norm Evolution per Hospital', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'figA4_gradient_norms.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figA4_gradient_norms.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure A5: Communication Cost per Round
    fig, ax = plt.subplots(figsize=(10, 5))

    total_comm = []
    for r in range(50):
        total = sum(detailed_results['communication_bytes'][name][r]
                   for name in detailed_results['communication_bytes']) / 1024  # KB
        total_comm.append(total)

    ax.bar(range(1, 51), total_comm, color='#3498db', edgecolor='black', alpha=0.7)
    ax.axhline(y=np.mean(total_comm), color='red', linestyle='--', label=f'Mean: {np.mean(total_comm):.1f} KB')
    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Communication (KB)', fontsize=11)
    ax.set_title('Total Communication Cost per Round', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'figA5_communication_cost.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figA5_communication_cost.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure A6: Learning Rate Comparison
    fig, ax = plt.subplots(figsize=(10, 5))

    for key, accs in lr_results.items():
        lr_val = key.replace('lr_', '')
        ax.plot(range(1, 31), accs, label=f'lr={lr_val}', lw=2)

    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Test Accuracy', fontsize=11)
    ax.set_title('Learning Rate Impact on Convergence', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.45, 0.65)

    plt.tight_layout()
    plt.savefig(output_dir / 'figA6_learning_rate.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figA6_learning_rate.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure A7: Batch Size Impact
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    bs_labels = [k.replace('bs_', '') for k in bs_results.keys()]
    final_accs = [bs_results[k]['accuracies'][-1] for k in bs_results]
    avg_times = [np.mean(bs_results[k]['times']) for k in bs_results]

    ax = axes[0]
    ax.bar(bs_labels, final_accs, color='#2ecc71', edgecolor='black')
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Final Accuracy', fontsize=11)
    ax.set_title('(a) Accuracy vs Batch Size', fontsize=12)
    ax.set_ylim(0.5, 0.65)
    for i, v in enumerate(final_accs):
        ax.text(i, v + 0.005, f'{v:.1%}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    ax = axes[1]
    ax.bar(bs_labels, [t*1000 for t in avg_times], color='#e74c3c', edgecolor='black')
    ax.set_xlabel('Batch Size', fontsize=11)
    ax.set_ylabel('Avg Time per Round (ms)', fontsize=11)
    ax.set_title('(b) Training Time vs Batch Size', fontsize=12)
    for i, v in enumerate(avg_times):
        ax.text(i, v*1000 + 0.5, f'{v*1000:.1f}', ha='center', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig(output_dir / 'figA7_batch_size.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figA7_batch_size.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Figure A8: Per-Client Accuracy Over Time
    fig, ax = plt.subplots(figsize=(10, 5))

    for name, data in detailed_results['per_client_per_round'].items():
        short_name = name.split('-')[1]
        accs = [d['accuracy'] if d['participated'] else None for d in data]
        valid_accs = [(i+1, a) for i, a in enumerate(accs) if a is not None]
        if valid_accs:
            rounds, acc_vals = zip(*valid_accs)
            ax.plot(rounds, acc_vals, 'o-', label=short_name, markersize=3, lw=1)

    ax.set_xlabel('Round', fontsize=11)
    ax.set_ylabel('Local Accuracy', fontsize=11)
    ax.set_title('Per-Hospital Local Accuracy Over Training', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.5, 0.75)

    plt.tight_layout()
    plt.savefig(output_dir / 'figA8_client_accuracy.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(output_dir / 'figA8_client_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()

    print(f"\nAll appendix figures saved to {output_dir}")

def save_appendix_results(detailed, lr, bs, output_dir: Path):
    """Save all results to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / 'appendix_detailed_results.json', 'w') as f:
        json.dump(detailed, f, indent=2)

    with open(output_dir / 'appendix_lr_results.json', 'w') as f:
        json.dump(lr, f, indent=2)

    with open(output_dir / 'appendix_bs_results.json', 'w') as f:
        json.dump(bs, f, indent=2)

    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    print("FL-EHDS Appendix Experiments")
    print("=" * 60)
    print(f"Started at: {datetime.now().isoformat()}")

    detailed = run_detailed_fl_experiment()
    lr = run_learning_rate_experiment()
    bs = run_batch_size_experiment()

    output_dir = Path(__file__).parent / "results_appendix"

    save_appendix_results(detailed, lr, bs, output_dir)
    generate_appendix_figures(detailed, lr, bs, output_dir)

    # Print summary
    print("\n" + "=" * 60)
    print("APPENDIX DATA SUMMARY")
    print("=" * 60)

    print("\nHospital Statistics:")
    for s in detailed['hospital_stats']:
        print(f"  {s['name']}: {s['n_samples']} samples, "
              f"age={s['mean_age']:.1f}, pos_rate={s['positive_rate']:.1%}")

    print("\nParticipation Rates:")
    for name in detailed['per_client_per_round']:
        data = detailed['per_client_per_round'][name]
        rate = sum(1 for d in data if d['participated']) / len(data)
        print(f"  {name.split('-')[1]}: {rate:.0%}")

    print("\n" + "=" * 60)
    print("EXPERIMENTS COMPLETED")
    print("=" * 60)
