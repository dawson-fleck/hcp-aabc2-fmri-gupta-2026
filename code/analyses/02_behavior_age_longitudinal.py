#!/usr/bin/env python3
"""Baseline-age and longitudinal-time associations with CARIT and FACENAME performance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from lib.analysis_utils import (  # noqa: E402
    CARIT_DVS, FACENAME_DVS, bh_fdr, coefficient_table, ensure_output_dir, fit_clustered_ols,
    fit_gee, joint_wald, matching_terms, read_analysis_table, stack_outcomes
)


def stacked_omnibus(df: pd.DataFrame, dv_map: dict[str, str], task: str):
    long = stack_outcomes(df, dv_map)
    formula = "z ~ 0 + C(DV_type) + C(DV_type):baseline_age_c + C(DV_type):time_years"
    gee = fit_gee(formula, long)
    ols = fit_clustered_ols(formula, long)
    rows = []
    for estimator, result in [("GEE robust", gee), ("Clustered OLS", ols)]:
        for effect, token in [("baseline_age", "baseline_age_c"), ("longitudinal_time", "time_years")]:
            terms = matching_terms(result, ["C(DV_type)", token])
            test = joint_wald(result, terms)
            rows.append({"task": task, "estimator": estimator, "effect": effect, **{k: test[k] for k in ["chi2", "df", "p"]}})
    return pd.DataFrame(rows), coefficient_table(gee, f"{task}_stacked_GEE")


def per_outcome_models(df: pd.DataFrame, dv_map: dict[str, str], task: str) -> pd.DataFrame:
    rows = []
    for label, col in dv_map.items():
        d = df[["SubjectID", col, "baseline_age_c", "time_years"]].dropna().copy()
        for estimator, fit in [
            ("GEE robust", fit_gee(f"Q('{col}') ~ baseline_age_c + time_years", d)),
            ("Clustered OLS", fit_clustered_ols(f"Q('{col}') ~ baseline_age_c + time_years", d)),
        ]:
            ci = fit.conf_int()
            for effect in ["baseline_age_c", "time_years"]:
                rows.append({
                    "task": task, "outcome": label, "column": col, "estimator": estimator, "effect": effect,
                    "N": len(d), "estimate_per_year": fit.params[effect], "SE": fit.bse[effect], "p": fit.pvalues[effect],
                    "CI95_low": ci.loc[effect, 0], "CI95_high": ci.loc[effect, 1],
                    "estimate_per_10_years": fit.params[effect] * 10,
                })
    out = pd.DataFrame(rows)
    for task_name in out["task"].unique():
        for estimator in out["estimator"].unique():
            for effect in ["baseline_age_c", "time_years"]:
                mask = (out.task == task_name) & (out.estimator == estimator) & (out.effect == effect)
                out.loc[mask, "q_BH"] = bh_fdr(out.loc[mask, "p"])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/behavior_age_longitudinal")
    args = parser.parse_args()

    df = read_analysis_table(args.input)
    outdir = ensure_output_dir(args.output_dir)
    omnibus_all, coef_all, per_all = [], [], []
    for task, mapping in [("CARIT", CARIT_DVS), ("FACENAME", FACENAME_DVS)]:
        omnibus, coef = stacked_omnibus(df, mapping, task)
        omnibus_all.append(omnibus)
        coef_all.append(coef)
        per_all.append(per_outcome_models(df, mapping, task))
    pd.concat(omnibus_all, ignore_index=True).to_csv(outdir / "behavior_omnibus_tests.csv", index=False)
    pd.concat(coef_all, ignore_index=True).to_csv(outdir / "behavior_stacked_coefficients.csv", index=False)
    pd.concat(per_all, ignore_index=True).to_csv(outdir / "behavior_per_outcome_models.csv", index=False)


if __name__ == "__main__":
    main()
