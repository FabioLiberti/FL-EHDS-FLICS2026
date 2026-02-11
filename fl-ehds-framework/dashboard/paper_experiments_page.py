"""
Paper Experiments page for FL-EHDS Streamlit dashboard.
Provides incremental experiment management for FLICS 2026 paper.
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime

import streamlit as st
import numpy as np
import pandas as pd

FRAMEWORK_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(FRAMEWORK_DIR))

from benchmarks.generate_paper_outputs import (
    load_all_checkpoints, compute_significance,
    generate_multi_dataset_table, generate_ablation_table,
    generate_attack_table, generate_figures, OUTPUT_DIR,
    ALGORITHMS, ALL_DATASETS, SEEDS,
    IMAGING_DATASETS as IMG_DS, TABULAR_DATASETS as TAB_DS,
)

ALL_DS_LIST = list(IMG_DS.keys()) + list(TAB_DS.keys())


def render_paper_experiments_tab():
    """Render the Paper Experiments tab with 3 sub-tabs."""
    st.header("Esperimenti Paper FLICS 2026")
    st.caption("Workflow incrementale con checkpoint/resume per generazione risultati paper")

    tab1, tab2, tab3 = st.tabs([
        "Status Dashboard",
        "Esegui Esperimenti",
        "Output Generati",
    ])

    with tab1:
        _render_status_tab()

    with tab2:
        _render_execute_tab()

    with tab3:
        _render_output_tab()


# ======================================================================
# Sub-tab 1: Status Dashboard
# ======================================================================

def _render_status_tab():
    """Show experiment completion status."""
    if st.button("Aggiorna Status", key="refresh_status"):
        st.rerun()

    try:
        checkpoints = load_all_checkpoints()
    except Exception as e:
        st.error(f"Errore caricamento checkpoint: {e}")
        return

    # P1.2 Multi-Dataset
    st.subheader("P1.2 Multi-Dataset FL Comparison")
    p12 = checkpoints.get("p12")
    total_p12 = len(ALGORITHMS) * len(ALL_DS_LIST) * len(SEEDS)

    if p12 and "completed" in p12:
        completed = p12["completed"]
        ok = {k: v for k, v in completed.items() if "error" not in v}
        err = {k: v for k, v in completed.items() if "error" in v}

        col1, col2, col3 = st.columns(3)
        col1.metric("Completati", f"{len(ok)}/{total_p12}")
        col2.metric("Errori", len(err))
        col3.metric("Progresso", f"{len(ok)/total_p12*100:.0f}%")

        # Completion matrix
        matrix_data = []
        for ds in ALL_DS_LIST:
            row = {"Dataset": ds}
            for algo in ALGORITHMS:
                seeds_done = sum(1 for s in SEEDS if f"{ds}_{algo}_{s}" in ok)
                if seeds_done == len(SEEDS):
                    row[algo] = "OK"
                elif seeds_done > 0:
                    row[algo] = f"{seeds_done}/{len(SEEDS)}"
                else:
                    row[algo] = "--"
            matrix_data.append(row)

        df = pd.DataFrame(matrix_data)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Best results
        st.caption("Migliori risultati per dataset")
        best_data = []
        for ds in ALL_DS_LIST:
            best_acc = 0
            best_algo = None
            for algo in ALGORITHMS:
                accs = []
                for seed in SEEDS:
                    rec = ok.get(f"{ds}_{algo}_{seed}", {})
                    fm = rec.get("final_metrics")
                    if fm:
                        accs.append(fm["accuracy"])
                if accs and np.mean(accs) > best_acc:
                    best_acc = np.mean(accs)
                    best_algo = algo
            if best_algo:
                f1s = [ok[f"{ds}_{best_algo}_{s}"]["final_metrics"]["f1"]
                       for s in SEEDS if f"{ds}_{best_algo}_{s}" in ok
                       and "final_metrics" in ok[f"{ds}_{best_algo}_{s}"]]
                aucs = [ok[f"{ds}_{best_algo}_{s}"]["final_metrics"]["auc"]
                        for s in SEEDS if f"{ds}_{best_algo}_{s}" in ok
                        and "final_metrics" in ok[f"{ds}_{best_algo}_{s}"]]
                best_data.append({
                    "Dataset": ds,
                    "Best Algorithm": best_algo,
                    "Accuracy": f"{best_acc*100:.1f}%",
                    "F1": f"{np.mean(f1s):.3f}" if f1s else "--",
                    "AUC": f"{np.mean(aucs):.3f}" if aucs else "--",
                })
        if best_data:
            st.dataframe(pd.DataFrame(best_data), use_container_width=True, hide_index=True)
    else:
        st.info("Nessun checkpoint P1.2 trovato. Esegui esperimenti multi-dataset.")

    # P1.3 Ablation
    st.subheader("P1.3 Studio Ablativo (chest_xray)")
    p13 = checkpoints.get("p13")

    if p13 and "completed" in p13:
        completed = p13["completed"]
        ok = {k: v for k, v in completed.items() if "error" not in v}

        groups = {"clip": 0, "epsilon": 0, "model": 0, "classweights": 0}
        for key in ok:
            for g in groups:
                if key.startswith(g):
                    groups[g] += 1

        cols = st.columns(4)
        cols[0].metric("Clip (C)", f"{groups['clip']}/9")
        cols[1].metric("Epsilon", f"{groups['epsilon']}/18")
        cols[2].metric("Model", f"{groups['model']}/6")
        cols[3].metric("Weights", f"{groups['classweights']}/6")

        st.metric("Totale", f"{len(ok)}/39")
    else:
        st.info("Nessun checkpoint P1.3 trovato.")

    # P2.2 Attack
    st.subheader("P2.2 Privacy Attack (DLG)")
    p22 = checkpoints.get("p22")

    if p22 and "completed" in p22:
        completed = p22["completed"]
        ok = {k: v for k, v in completed.items() if "error" not in v}

        st.metric("Completati", f"{len(ok)}/12")

        attack_data = []
        for eps_label in ["inf", "10.0", "5.0", "1.0"]:
            vals = [v["reconstruction_mse"] for k, v in ok.items()
                    if k.startswith(f"eps_{eps_label}") and "reconstruction_mse" in v]
            if vals:
                eps_disp = "inf (no DP)" if eps_label == "inf" else f"eps={eps_label}"
                attack_data.append({
                    "Epsilon": eps_disp,
                    "MSE": f"{np.mean(vals):.3f}",
                    "Std": f"{np.std(vals):.3f}",
                    "Protection": "None" if eps_label == "inf" else
                                  ("Low" if eps_label == "10.0" else
                                   ("Medium" if eps_label == "5.0" else "High")),
                })
        if attack_data:
            st.dataframe(pd.DataFrame(attack_data), use_container_width=True, hide_index=True)
    else:
        st.info("Nessun checkpoint P2.2 trovato.")

    # P2.1 Significance
    p21 = checkpoints.get("p21")
    if p21:
        st.subheader("P2.1 Significativita' Statistica")
        n_comparisons = sum(len(v) for v in p21.values() if isinstance(v, dict))
        st.metric("Confronti calcolati", n_comparisons)


# ======================================================================
# Sub-tab 2: Execute Experiments
# ======================================================================

def _render_execute_tab():
    """Controls to launch experiments."""
    st.subheader("Esegui Esperimenti")
    st.caption("Gli esperimenti vengono lanciati in background tramite subprocess.")

    block = st.selectbox("Blocco sperimentale", [
        "P1.2 Multi-Dataset",
        "P1.3 Ablation Study",
        "P2.2 Privacy Attack",
    ], key="exp_block")

    extra_args = []

    if block == "P1.2 Multi-Dataset":
        col1, col2 = st.columns(2)
        with col1:
            ds_choice = st.selectbox("Dataset", ["Tutti"] + ALL_DS_LIST, key="exp_ds")
        with col2:
            algo_choice = st.selectbox("Algoritmo", ["Tutti"] + ALGORITHMS, key="exp_algo")

        extra_args = ["--only", "p12", "--resume"]
        if ds_choice != "Tutti":
            extra_args += ["--dataset", ds_choice]
        if algo_choice != "Tutti":
            extra_args += ["--algo", algo_choice]

    elif block == "P1.3 Ablation Study":
        extra_args = ["--only", "p13", "--resume"]
        st.info("Ablation study su chest_xray: clip, epsilon, model, class weights")

    elif block == "P2.2 Privacy Attack":
        extra_args = ["--only", "p22"]
        st.info("DLG gradient inversion attack (~60 secondi)")

    cmd = ["python", "-m", "benchmarks.run_paper_experiments"] + extra_args
    cmd_str = " ".join(cmd)

    st.code(cmd_str, language="bash")

    if st.button("Avvia Esperimento", key="launch_exp", type="primary"):
        try:
            process = subprocess.Popen(
                cmd,
                cwd=str(FRAMEWORK_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            st.success(f"Esperimento avviato (PID: {process.pid})")
            st.info("L'esperimento prosegue in background. Aggiorna lo Status Dashboard per monitorare i risultati.")

            # Show initial output
            with st.expander("Output iniziale (primi 30 secondi)", expanded=True):
                output_placeholder = st.empty()
                output_lines = []
                import time
                start = time.time()
                while time.time() - start < 30:
                    line = process.stdout.readline()
                    if not line:
                        if process.poll() is not None:
                            break
                        continue
                    output_lines.append(line.rstrip())
                    if len(output_lines) > 50:
                        output_lines = output_lines[-50:]
                    output_placeholder.code("\n".join(output_lines))

                if process.poll() is not None:
                    st.success(f"Processo terminato (exit code: {process.returncode})")
                else:
                    st.info("Il processo continua in background...")

        except Exception as e:
            st.error(f"Errore avvio: {e}")


# ======================================================================
# Sub-tab 3: Output Generated
# ======================================================================

def _render_output_tab():
    """Show and regenerate output files."""
    st.subheader("Output Generati")

    if st.button("Genera/Aggiorna Output", key="gen_output", type="primary"):
        with st.spinner("Generazione in corso..."):
            try:
                checkpoints = load_all_checkpoints()
                p12 = checkpoints.get("p12")
                p13 = checkpoints.get("p13")
                p22 = checkpoints.get("p22")

                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                generated = []

                sig = {}
                if p12:
                    try:
                        sig = compute_significance(p12)
                    except Exception:
                        sig = checkpoints.get("p21") or {}

                if p12:
                    tex = generate_multi_dataset_table(p12, sig)
                    (OUTPUT_DIR / "table_multi_dataset.tex").write_text(tex)
                    generated.append("table_multi_dataset.tex")

                if p13:
                    tex = generate_ablation_table(p13)
                    (OUTPUT_DIR / "table_ablation.tex").write_text(tex)
                    generated.append("table_ablation.tex")

                if p22:
                    tex = generate_attack_table(p22)
                    (OUTPUT_DIR / "table_attack.tex").write_text(tex)
                    generated.append("table_attack.tex")

                try:
                    n = generate_figures(p12 or {}, p13 or {}, p22 or {}, OUTPUT_DIR)
                    if n and n > 0:
                        generated.append(f"{n} figure")
                except Exception as e:
                    st.warning(f"Errore figure: {e}")

                if generated:
                    st.success(f"Generati: {', '.join(generated)}")
                else:
                    st.warning("Nessun dato disponibile per generare output.")
            except Exception as e:
                st.error(f"Errore: {e}")

    # List existing files
    st.subheader("File disponibili")
    if OUTPUT_DIR.exists():
        tex_files = []
        img_files = []
        other_files = []

        for f in sorted(OUTPUT_DIR.iterdir()):
            if f.name.startswith("checkpoint"):
                continue
            if f.suffix == ".tex":
                tex_files.append(f)
            elif f.suffix in (".png", ".pdf"):
                img_files.append(f)
            elif f.suffix == ".json":
                continue
            else:
                other_files.append(f)

        # LaTeX tables
        if tex_files:
            st.subheader("Tabelle LaTeX")
            for f in tex_files:
                size = f.stat().st_size
                size_str = f"{size / 1024:.1f} KB" if size > 1024 else f"{size} B"
                with st.expander(f"{f.name} ({size_str})"):
                    st.code(f.read_text(), language="latex")

        # Figures
        if img_files:
            st.subheader("Figure")
            png_files = [f for f in img_files if f.suffix == ".png"]
            pdf_files = [f for f in img_files if f.suffix == ".pdf"]

            if png_files:
                cols = st.columns(min(len(png_files), 2))
                for i, f in enumerate(png_files):
                    with cols[i % 2]:
                        st.image(str(f), caption=f.name, use_container_width=True)

            if pdf_files:
                st.caption("File PDF (non visualizzabili inline):")
                for f in pdf_files:
                    size = f.stat().st_size
                    st.text(f"  {f.name} ({size / 1024:.1f} KB)")

        if not tex_files and not img_files:
            st.info("Nessun file output trovato. Genera gli output dopo aver completato esperimenti.")
    else:
        st.info("La cartella output non esiste ancora.")
