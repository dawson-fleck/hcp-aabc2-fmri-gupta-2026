#!/usr/bin/env python3
"""Baseline-age and longitudinal-time associations with task-specific network Full amplitude."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from lib.analysis_utils import (  # noqa: E402
    NETWORK_COLUMNS, bh_fdr, ensure_output_dir, fit_gee, joint_wald, matching_terms,
    read_analysis_table, stack_networks
)


def run_task(df: pd.DataFrame, task: str, mapping: dict[str, str]):
    long = stack_networks(df, mapping)
    formula = "z_amplitude ~ 0 + C(Network) + C(Network):baseline_age_c + C(Network):time_years"
    gee = fit_gee(formula, long)
    omnibus = []
    for effect, token in [("baseline_age", "baseline_age_c"), ("longitudinal_time", "time_years")]:
        terms = matching_terms(gee, ["C(Network)", token])
        test = joint_wald(gee, terms)
        omnibus.append({"task": task, "effect": effect, **{k: test[k] for k in ["chi2", "df", "p"]}})

    per = []
    for network, col in mapping.items():
        d = df[["SubjectID", col, "baseline_age_c", "time_years"]].dropna().copy()
        fit = fit_gee(f"Q('{col}') ~ baseline_age_c + time_years", d)
        ci = fit.conf_int()
        for effect in ["baseline_age_c", "time_years"]:
            per.append({
                "task": task, "network": network, "effect": effect, "N": len(d),
                "estimate_per_year": fit.params[effect], "SE": fit.bse[effect], "p": fit.pvalues[effect],
                "CI95_low": ci.loc[effect, 0], "CI95_high": ci.loc[effect, 1],
                "estimate_per_10_years": fit.params[effect] * 10,
            })
    per = pd.DataFrame(per)
    for effect in ["baseline_age_c", "time_years"]:
        mask = per.effect.eq(effect)
        per.loc[mask, "q_BH"] = bh_fdr(per.loc[mask, "p"])
    return pd.DataFrame(omnibus), per


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/network_age")
    args = parser.parse_args()
    df = read_analysis_table(args.input)
    outdir = ensure_output_dir(args.output_dir)

    omnibus, per = [], []
    for task, mapping in NETWORK_COLUMNS.items():
        a, b = run_task(df, task, mapping)
        omnibus.append(a); per.append(b)
    pd.concat(omnibus, ignore_index=True).to_csv(outdir / "network_age_omnibus.csv", index=False)
    pd.concat(per, ignore_index=True).to_csv(outdir / "network_age_per_network.csv", index=False)


if __name__ == "__main__":
    main()
