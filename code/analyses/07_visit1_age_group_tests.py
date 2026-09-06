#!/usr/bin/env python3
"""Visit-1 Younger-versus-Older network Full-amplitude comparisons."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from lib.analysis_utils import NETWORK_COLUMNS, ensure_output_dir, holm, read_analysis_table  # noqa: E402


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    nx, ny = len(x), len(y)
    s_pooled = math.sqrt(((nx - 1) * x.var(ddof=1) + (ny - 1) * y.var(ddof=1)) / (nx + ny - 2))
    d = (x.mean() - y.mean()) / s_pooled
    correction = 1 - 3 / (4 * (nx + ny) - 9)
    return correction * d


def welch_df(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    nx, ny = len(x), len(y)
    num = (vx / nx + vy / ny) ** 2
    den = (vx ** 2) / (nx ** 2 * (nx - 1)) + (vy ** 2) / (ny ** 2 * (ny - 1))
    return num / den


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/visit1_age_groups")
    parser.add_argument("--younger-min", type=float, default=37)
    parser.add_argument("--younger-max", type=float, default=59)
    parser.add_argument("--older-min", type=float, default=70)
    parser.add_argument("--older-max", type=float, default=88)
    args = parser.parse_args()

    df = read_analysis_table(args.input)
    v1 = df.loc[df["Visit"].eq("V1")].copy()
    age = pd.to_numeric(v1["baseline_age_v1"], errors="coerce")
    v1["AgeGroup"] = np.select(
        [age.between(args.younger_min, args.younger_max), age.between(args.older_min, args.older_max)],
        ["Younger", "Older"], default="Middle"
    )

    rows = []
    for task, mapping in NETWORK_COLUMNS.items():
        for network, col in mapping.items():
            y = pd.to_numeric(v1.loc[v1.AgeGroup.eq("Younger"), col], errors="coerce").dropna().to_numpy()
            o = pd.to_numeric(v1.loc[v1.AgeGroup.eq("Older"), col], errors="coerce").dropna().to_numpy()
            t, p = ttest_ind(y, o, equal_var=False)
            rows.append({
                "task": task, "network": network, "N_younger": len(y), "N_older": len(o),
                "mean_younger": y.mean(), "mean_older": o.mean(), "percent_lower_older": 100 * (y.mean() - o.mean()) / y.mean(),
                "t": t, "df": welch_df(y, o), "p": p, "Hedges_g": hedges_g(y, o),
            })
    out = pd.DataFrame(rows)
    out["p_Holm"] = holm(out["p"])
    outdir = ensure_output_dir(args.output_dir)
    out.to_csv(outdir / "visit1_younger_older_network_tests.csv", index=False)


if __name__ == "__main__":
    main()
