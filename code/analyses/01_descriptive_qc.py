#!/usr/bin/env python3
"""Descriptive statistics, network correlations, and imaging/motion QC summaries."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from lib.analysis_utils import (  # noqa: E402
    CARIT_DVS, FACENAME_DVS, NETWORK_COLUMNS, ensure_output_dir, read_analysis_table, require_columns
)


def summarize_numeric(df: pd.DataFrame, columns: list[str], group: str | None = None) -> pd.DataFrame:
    rows = []
    groups = [("All", df)] if group is None else list(df.groupby(group, observed=True))
    for label, g in groups:
        for col in columns:
            x = pd.to_numeric(g[col], errors="coerce")
            rows.append({
                "group": label, "variable": col, "N": x.notna().sum(), "mean": x.mean(), "SD": x.std(ddof=1),
                "min": x.min(), "max": x.max(),
            })
    return pd.DataFrame(rows)


def network_correlations(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, mapping in NETWORK_COLUMNS.items():
        cols = list(mapping.values())
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                x = pd.to_numeric(df[cols[i]], errors="coerce")
                y = pd.to_numeric(df[cols[j]], errors="coerce")
                ok = x.notna() & y.notna()
                r, p = pearsonr(x[ok], y[ok])
                rows.append({"task": task, "network_1": list(mapping.keys())[i], "network_2": list(mapping.keys())[j], "N": ok.sum(), "r": r, "p": p})
    return pd.DataFrame(rows)


def motion_correlations(df: pd.DataFrame) -> pd.DataFrame:
    if "RelativeRMS_mean" not in df:
        return pd.DataFrame()
    rows = []
    motion = pd.to_numeric(df["RelativeRMS_mean"], errors="coerce")
    for task, mapping in NETWORK_COLUMNS.items():
        for network, col in mapping.items():
            y = pd.to_numeric(df[col], errors="coerce")
            ok = motion.notna() & y.notna()
            if ok.sum() > 2:
                r, p = pearsonr(motion[ok], y[ok])
                rows.append({"task": task, "network": network, "N": ok.sum(), "r": r, "p": p})
    return pd.DataFrame(rows)


def iqr_flags(df: pd.DataFrame) -> pd.DataFrame:
    variables = [*CARIT_DVS.values(), *FACENAME_DVS.values(), *NETWORK_COLUMNS["CARIT"].values(), *NETWORK_COLUMNS["FACENAME"].values()]
    rows = []
    for col in variables:
        x = pd.to_numeric(df[col], errors="coerce")
        q1, q3 = x.quantile([0.25, 0.75])
        iqr = q3 - q1
        for mult in (1.5, 3.0):
            lo, hi = q1 - mult * iqr, q3 + mult * iqr
            n = ((x < lo) | (x > hi)).sum()
            rows.append({"variable": col, "criterion_IQR": mult, "lower": lo, "upper": hi, "N_flagged": int(n)})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/descriptive_qc")
    args = parser.parse_args()

    df = read_analysis_table(args.input)
    require_columns(df, [*CARIT_DVS.values(), *FACENAME_DVS.values(), *NETWORK_COLUMNS["CARIT"].values(), *NETWORK_COLUMNS["FACENAME"].values()], "analysis table")
    out = ensure_output_dir(args.output_dir)

    numeric = ["baseline_age_v1", "time_years", *CARIT_DVS.values(), *FACENAME_DVS.values(), *NETWORK_COLUMNS["CARIT"].values(), *NETWORK_COLUMNS["FACENAME"].values()]
    summarize_numeric(df, numeric).to_csv(out / "descriptive_statistics.csv", index=False)
    if "Visit" in df:
        summarize_numeric(df, numeric, "Visit").to_csv(out / "descriptive_by_visit.csv", index=False)
    network_correlations(df).to_csv(out / "network_correlations.csv", index=False)
    motion_correlations(df).to_csv(out / "motion_network_correlations.csv", index=False)
    iqr_flags(df).to_csv(out / "iqr_flags.csv", index=False)

    qc = {
        "participants": df["SubjectID"].nunique(),
        "participant_visits": len(df),
        "qc_issue_visits": int(pd.to_numeric(df.get("qc_any"), errors="coerce").eq(1).sum()) if "qc_any" in df else np.nan,
        "qc_issue_participants": int(df.loc[pd.to_numeric(df.get("qc_any"), errors="coerce").eq(1), "SubjectID"].nunique()) if "qc_any" in df else np.nan,
    }
    pd.DataFrame([qc]).to_csv(out / "qc_counts.csv", index=False)


if __name__ == "__main__":
    main()
