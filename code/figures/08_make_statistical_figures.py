#!/usr/bin/env python3
"""Generate the statistical plots used for longitudinal behavior and FPN age moderation.

Cortical surface maps are not generated here because they require the local
AABC parcel files and Connectome Workbench. See docs/surface_map_workflow.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from lib.analysis_utils import CARIT_DVS, FACENAME_DVS, NETWORK_COLUMNS, ensure_output_dir, fit_gee, read_analysis_table, stack_outcomes  # noqa: E402


def longitudinal_effect_plot(effect_csv: str | Path, output: Path) -> None:
    d = pd.read_csv(effect_csv)
    d = d.loc[(d.estimator == "GEE robust") & (d.effect == "time_years")].copy()
    labels = d["task"] + " | " + d["outcome"]
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, max(5, 0.38 * len(d))))
    ax.errorbar(d["estimate_per_year"], y,
                xerr=[d["estimate_per_year"] - d["CI95_low"], d["CI95_high"] - d["estimate_per_year"]],
                fmt="o", capsize=3)
    ax.axvline(0, linestyle="--", linewidth=1)
    ax.set_yticks(y); ax.set_yticklabels(labels)
    ax.set_xlabel("Change per year")
    ax.set_title("Longitudinal task-performance effects")
    fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)


def observed_visit_plot(df: pd.DataFrame, output: Path) -> None:
    panels = [("CARIT", CARIT_DVS), ("FACENAME", FACENAME_DVS)]
    for task, dvs in panels:
        n = len(dvs)
        fig, axes = plt.subplots(n, 1, figsize=(7, 2.4 * n), sharex=True)
        axes = np.atleast_1d(axes)
        for ax, (label, col) in zip(axes, dvs.items()):
            g = df.groupby("Visit", observed=True)[col].agg(["mean", "std", "count"]).reindex(["V1", "V2", "V3", "V4"])
            se = g["std"] / np.sqrt(g["count"])
            ax.errorbar(g.index, g["mean"], yerr=1.96 * se, marker="o", capsize=3)
            ax.set_ylabel(label)
        axes[-1].set_xlabel("Visit")
        fig.suptitle(f"Observed {task} performance across visits")
        fig.tight_layout(); fig.savefig(output.with_name(f"{output.stem}_{task.lower()}{output.suffix}"), dpi=300); plt.close(fig)


def fpn_age_slope_plot(df: pd.DataFrame, output: Path) -> None:
    selections = [
        ("CARIT", CARIT_DVS, "d_prime"),
        ("FACENAME", FACENAME_DVS, "Percent Correct"),
        ("CARIT", CARIT_DVS, "False Alarm Rate"),
        ("FACENAME", FACENAME_DVS, "Percent Omission"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, (task, dvs, chosen) in zip(axes.ravel(), selections):
        fpn = NETWORK_COLUMNS[task]["FPN"]
        long = stack_outcomes(df, dvs).merge(df[["SubjectID", "Visit", fpn]], on=["SubjectID", "Visit"], how="left").dropna(subset=[fpn])
        fit = fit_gee(f"z ~ C(DV_type) * Q('{fpn}') * baseline_age_c + time_years", long)
        ages = np.linspace(df["baseline_age_v1"].min(), df["baseline_age_v1"].max(), 80)
        age_mean = df["baseline_age_v1"].drop_duplicates().mean()
        level = chosen
        ref = list(dvs.keys())[0]
        base_fpn = fit.params.get(f"Q('{fpn}')", 0.0)
        base_age_int = fit.params.get(f"Q('{fpn}'):baseline_age_c", 0.0)
        dv_fpn = 0.0; dv_age_int = 0.0
        if level != ref:
            for name, val in fit.params.items():
                if f"[T.{level}]" in name and f"Q('{fpn}')" in name and "baseline_age_c" not in name:
                    dv_fpn += val
                if f"[T.{level}]" in name and f"Q('{fpn}')" in name and "baseline_age_c" in name:
                    dv_age_int += val
        slopes = base_fpn + dv_fpn + (base_age_int + dv_age_int) * (ages - age_mean)
        ax.plot(ages, slopes)
        ax.axhline(0, linestyle="--", linewidth=1)
        ax.set_title(f"{task}: {level}")
        ax.set_xlabel("Baseline age"); ax.set_ylabel("Standardized FPN-performance slope")
    fig.tight_layout(); fig.savefig(output, dpi=300); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Prepared analysis CSV or source workbook")
    parser.add_argument("--behavior-models", help="behavior_per_outcome_models.csv from script 02")
    parser.add_argument("--output-dir", default="outputs/figures")
    args = parser.parse_args()
    out = ensure_output_dir(args.output_dir)
    df = read_analysis_table(args.input)
    if args.behavior_models:
        longitudinal_effect_plot(args.behavior_models, out / "longitudinal_effects.png")
    observed_visit_plot(df, out / "observed_visits.png")
    fpn_age_slope_plot(df, out / "fpn_age_moderation.png")


if __name__ == "__main__":
    main()
