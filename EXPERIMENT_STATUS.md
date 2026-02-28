# FL-EHDS FLICS2026 - Experiment Status

> File di interscambio tra MacBook Pro M1 e MacBook Air M3.
> Aggiornato automaticamente dal pre-commit hook (git_hooks/pre-commit).

**Last update:** 2026-02-28T09:34:16 CET (192)

---

## Machines

| Machine | Ruolo | Stato attuale |
|---|---|---|
| MacBook Pro M1 | Centrale di processo, imaging pesante | Libero |
| MacBook Air M3 | Esperimenti tabular + imaging leggero | Libero (cascade completata) |
| RunPod (GPU) | Imaging scalability | In corso: Imaging Scalability (54 exp) |
| Google Colab | Imaging opt-out | In corso: Imaging Opt-out (~154 exp, 8+ completati) |

---

## Experiment Inventory

### TABULAR (MacBook Air M3 + MacBook Pro M1)

| Experiment | Script | Checkpoint | N. exp | Status |
|---|---|---|---|---|
| Baseline (optimized) | `run_tabular_optimized.py` | `paper_results_tabular/checkpoint_tabular.json` | 105 | DONE |
| Seeds x10 | `run_tabular_seeds10.py` | `paper_results_tabular/checkpoint_seeds10.json` | 105 | DONE |
| Differential Privacy | `run_tabular_dp.py` | `paper_results_tabular/checkpoint_dp.json` | 180 | DONE |
| Opt-out GDPR Art.71 | `run_tabular_optout.py` | `paper_results_tabular/checkpoint_optout.json` | 225 | DONE |
| Deep MLP | `run_tabular_deep_mlp.py` | `paper_results_tabular/checkpoint_deep_mlp.json` | 70 | DONE |
| Scalability K=50,100 | `run_scalability_sweep.py` | `paper_results_tabular/checkpoint_scalability.json` | 84 | DONE |
| Scalability + DP | `run_scalability_dp.py` | `paper_results_tabular/checkpoint_scalability_dp.json` | 54 | DONE |
| Scalability DP (CV) | `run_scalability_dp_cv.py` | `paper_results_tabular/checkpoint_scalability_dp_cv.json` | 18 | DONE |
| RDP vs Naive Comp. | `run_rdp_comparison.py` | `paper_results_tabular/rdp_comparison_results.json` | 1 (analytical) | DONE |
| Byzantine + DP | `run_byzantine_dp.py` | `paper_results_tabular/byzantine_dp_v2_results.json` | 198 | DONE |
| DP Per-class DEI | `run_dp_per_class.py` | `paper_results_tabular/dp_per_class_results.json` | 80 | DONE |
| Confusion Matrix BC | `run_confusion_matrix_bc.py` | `paper_results_tabular/checkpoint_confusion_bc.json` | 40 | DONE |
| Local vs Federated | `run_local_vs_federated.py` | `results/local_vs_federated/checkpoint_local_vs_fed.json` | 3 seeds | DONE |
| Epochs Sweep | `run_tabular_epochs_sweep.py` | `paper_results_tabular/checkpoint_epochs_sweep.json` | 140 | DONE |
| Top-K PTB-XL | `run_topk_ptbxl.py` | `paper_results_tabular/checkpoint_topk_ptbxl.json` | 9 | DONE |

**Tabular total: ~1,312 experiments - ALL COMPLETE**

### IMAGING (MacBook Pro M1 + RunPod + Colab)

| Experiment | Script | Checkpoint | N. exp | Status |
|---|---|---|---|---|
| CNN baseline (3 seeds) | `run_imaging_experiments.py` | `checkpoint_imaging_cnn.json` | 27 | DONE |
| Multidataset (5 seeds) | `run_imaging_multi.py` | `paper_results/checkpoint_p12_multidataset.json` | 129 | DONE |
| Seeds5 expansion | `run_imaging_seeds5.py` | `paper_results_delta/checkpoint_imaging_seeds5.json` | 24 | DONE |
| Imaging DP | `run_imaging_dp.py` | `checkpoint_imaging_dp.json` | 108 | DONE |
| Completion (delta) | `run_imaging_completion.py` | `paper_results_delta/checkpoint_completion.json` | 8 | DONE |
| Confusion Matrix BT | `run_confusion_matrix_bt.py` | `paper_results_delta/checkpoint_confusion_bt.json` | 6 | DONE |
| Chest X-ray extended | `run_imaging_chest.py` | `paper_results_delta/checkpoint_chest_extended.json` | 6 | DONE |
| Confusion Matrix Chest | `run_confusion_matrix_chest.py` | `paper_results_delta/checkpoint_confusion_chest.json` | 6 | DONE |
| Significance tests | -- | `paper_results/checkpoint_p21_significance.json` | 5 | DONE |
| Attack robustness | -- | `paper_results/checkpoint_p22_attack.json` | 4 | DONE |
| Imaging Opt-out | `run_imaging_optout.py` | (Colab/RunPod) | ~154 | IN PROGRESS (Colab) |
| Imaging Scalability | `run_imaging_scalability.py` | (RunPod) | 54 | IN PROGRESS (RunPod) |

**Imaging done: ~323 experiments | In progress: ~208 experiments**

### OTHER / APPENDIX

| Experiment | Checkpoint | N. exp | Status |
|---|---|---|---|
| Appendix BS sweep | `results_appendix/appendix_bs_results.json` | 5 | DONE |
| Appendix LR sweep | `results_appendix/appendix_lr_results.json` | 5 | DONE |
| Appendix detailed | `results_appendix/appendix_detailed_results.json` | 7 | DONE |
| Extended results | `results_extended/extended_results.json` | 19 | DONE |
| Heterogeneity | `results_heterogeneity/heterogeneity_metrics.json` | 5 | DONE |
| Opt-out impact | `results_optout/optout_impact_results.json` | 11 | DONE |
| Comm costs | `paper_results/checkpoint_comm_costs.json` | analytical | DONE |

---

## Run History

| Version | Date | Machine | Experiments | Time |
|---|---|---|---|---|
| v12.1 | 2026-02-25 | MacBook Air M3 | 685 tabular (baseline+seeds10+dp+optout+deep_mlp) | ~6h |
| v12.2 | 2026-02-25 | MacBook Air M3 | 139 supplementary (RDP+scalability+scalability_dp) | 33 min |
| v12.3 | 2026-02-26 | MacBook Air M3 | 296 priority2 (byzantine+dp_per_class+scal_dp_cv+local_vs_fed) | 10h 12m |
| v12.5 | 2026-02-27 | MacBook Air M3 | 24 imaging seeds5 (789, 999) | 2h 39m |
| v12.6 | 2026-02-27 | MacBook Air M3 | 155 remaining (epochs_sweep+topk+confusion_chest) | 4h 03m |
| -- | 2026-02-27 | RunPod | 54 imaging scalability (K=10,20) | In corso |
| -- | 2026-02-27 | Colab | ~154 imaging opt-out (GDPR Art.71) | In corso |

---

## Remaining Gaps

| Priority | Experiment | N. exp | Where | Status |
|---|---|---|---|---|
| MED | Imaging Opt-out (GDPR Art.71) | ~154 | Colab | In corso |
| MED | Imaging Scalability (K=10,20) | 54 | RunPod | In corso |
| LOW | Byzantine+DP imaging | -- | -- | Non prioritario |
| LOW | Per-class DEI imaging | -- | -- | Non prioritario |

---

## Quick Sync Protocol

```bash
# PUSH (dal Mac che esegue esperimenti):
bash sync_push.sh "v12.X descrizione"   # commit completo + aggiorna status
bash sync_push.sh                        # solo sync status (auto-commit)

# PULL (dall'altro Mac per monitorare):
bash sync_pull.sh                        # pull + dashboard stato

# Git hooks (automatici, trasparenti):
# pre-commit: aggiorna EXPERIMENT_STATUS.md leggendo i checkpoint
# post-merge: mostra dashboard dopo git pull
# Setup (una volta per macchina): git config core.hooksPath git_hooks
```

> Gli script e i git hooks leggono automaticamente i checkpoint JSON e aggiornano
> EXPERIMENT_STATUS.md. Non serve editing manuale.

---

## Datasets Available

| Dataset | Tabular/Imaging | MacBook Pro M1 | MacBook Air M3 | Colab/RunPod |
|---|---|---|---|---|
| Breast Cancer Wisconsin | Tabular | yes | yes | yes (sklearn) |
| Cardiovascular Disease | Tabular | yes | yes | yes (kagglehub) |
| PTB-XL ECG | Tabular | yes | yes | needs download |
| Heart Disease | Tabular | yes | yes | yes (sklearn) |
| Diabetes | Tabular | yes | yes | yes |
| Brain Tumor | Imaging | yes | yes | needs download |
| Skin Cancer | Imaging | yes | yes | needs download |
| Chest X-ray | Imaging | yes | yes | needs download |
