#!/usr/bin/env python3
"""Repeated-measures network-performance models and baseline-age moderation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from lib.analysis_utils import (  # noqa: E402
    CARIT_DVS, FACENAME_DVS, NETWORK_COLUMNS, ensure_output_dir, fit_best_mixedlm, fit_clustered_ols,
    fit_gee, joint_wald, likelihood_ratio_test, linear_combination, matching_terms, read_analysis_table,
    stack_outcomes
)


def add_network_to_long(df: pd.DataFrame, long: pd.DataFrame, network_col: str) -> pd.DataFrame:
    net = df[["SubjectID", "Visit", network_col]].copy()
    net[network_col] = pd.to_numeric(net[network_col], errors="coerce")
    return long.merge(net, on=["SubjectID", "Visit"], how="left").dropna(subset=[network_col])


def mixed_network_model(df: pd.DataFrame, task: str, dvs: dict[str, str], network: str, network_col: str):
    long = add_network_to_long(df, stack_outcomes(df, dvs), network_col)

    f2 = f"z ~ C(DV_type) * Q('{network_col}') + C(DV_type) * baseline_age_c + time_years"
    f2_reduced = f"z ~ C(DV_type) + Q('{network_col}') + C(DV_type) * baseline_age_c + time_years"
    full2, log_full2, opt2 = fit_best_mixedlm(f2, long, reml=False)
    red2, log_red2, _ = fit_best_mixedlm(f2_reduced, long, reml=False)
    lrt2 = likelihood_ratio_test(full2, red2)

    f3 = f"z ~ C(DV_type) * Q('{network_col}') * baseline_age_c + time_years"
    f3_reduced = (
        f"z ~ C(DV_type) * Q('{network_col}') + C(DV_type) * baseline_age_c "
        f"+ Q('{network_col}') * baseline_age_c + time_years"
    )
    full3, log_full3, opt3 = fit_best_mixedlm(f3, long, reml=False)
    red3, log_red3, _ = fit_best_mixedlm(f3_reduced, long, reml=False)
    lrt3 = likelihood_ratio_test(full3, red3)
    reml3, log_reml3, opt_reml = fit_best_mixedlm(f3, long, reml=True)

    levels = list(dvs.keys())
    reference = levels[0]
    base = f"Q('{network_col}'):baseline_age_c"
    local = []
    for dv in levels:
        terms = [base]
        if dv != reference:
            interaction = f"C(DV_type)[T.{dv}]:Q('{network_col}'):baseline_age_c"
            if interaction not in reml3.fe_params.index:
                interaction = f"Q('{network_col}'):C(DV_type)[T.{dv}]:baseline_age_c"
            terms.append(interaction)
        est = linear_combination(reml3, terms)
        local.append({"task": task, "network": network, "outcome": dv, **est})

    tests = pd.DataFrame([
        {"task": task, "network": network, "effect": "DV_type x network", **lrt2},
        {"task": task, "network": network, "effect": "DV_type x network x baseline_age", **lrt3},
    ])
    logs = pd.concat([log_full2.assign(model="two_way_full"), log_red2.assign(model="two_way_reduced"),
                      log_full3.assign(model="three_way_full"), log_red3.assign(model="three_way_reduced"),
                      log_reml3.assign(model="three_way_REML")], ignore_index=True)
    logs["task"] = task; logs["network"] = network
    return tests, pd.DataFrame(local), logs


def robust_fpn(df: pd.DataFrame, task: str, dvs: dict[str, str], fpn_col: str):
    long = add_network_to_long(df, stack_outcomes(df, dvs), fpn_col)
    formula = f"z ~ C(DV_type) * Q('{fpn_col}') * baseline_age_c + time_years"
    rows_test, rows_local = [], []
    levels = list(dvs.keys())
    ref = levels[0]

    estimators = [
        ("GEE robust", fit_gee(formula, long, "robust")),
        ("GEE bias-reduced", fit_gee(formula, long, "bias_reduced")),
        ("Clustered OLS", fit_clustered_ols(formula, long)),
    ]
    for label, fit in estimators:
        two = matching_terms(fit, ["C(DV_type)", f"Q('{fpn_col}')"], ["baseline_age_c"])
        three = matching_terms(fit, ["C(DV_type)", f"Q('{fpn_col}')", "baseline_age_c"])
        for effect, terms in [("DV_type x FPN", two), ("DV_type x FPN x baseline_age", three)]:
            stat = joint_wald(fit, terms)
            rows_test.append({"task": task, "estimator": label, "effect": effect, **{k: stat[k] for k in ["chi2", "df", "p"]}})

        base = f"Q('{fpn_col}'):baseline_age_c"
        for dv in levels:
            terms = [base]
            if dv != ref:
                interaction = f"C(DV_type)[T.{dv}]:Q('{fpn_col}'):baseline_age_c"
                if interaction not in fit.params.index:
                    interaction = f"Q('{fpn_col}'):C(DV_type)[T.{dv}]:baseline_age_c"
                terms.append(interaction)
            # linear_combination expects fixed-effect covariance; GEE/OLS also expose compatible params/cov.
            class Wrapper:
                pass
            w = Wrapper(); w.fe_params = fit.params; w.params = fit.params; w.cov_params = fit.cov_params
            est = linear_combination(w, terms)
            rows_local.append({"task": task, "estimator": label, "outcome": dv, **est})
    return pd.DataFrame(rows_test), pd.DataFrame(rows_local)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/age_moderation")
    args = parser.parse_args()
    df = read_analysis_table(args.input)
    out = ensure_output_dir(args.output_dir)

    tests, local, logs, robust_tests, robust_local = [], [], [], [], []
    for task, dvs in [("CARIT", CARIT_DVS), ("FACENAME", FACENAME_DVS)]:
        for network, col in NETWORK_COLUMNS[task].items():
            a, b, c = mixed_network_model(df, task, dvs, network, col)
            tests.append(a); local.append(b); logs.append(c)
        a, b = robust_fpn(df, task, dvs, NETWORK_COLUMNS[task]["FPN"])
        robust_tests.append(a); robust_local.append(b)

    pd.concat(tests, ignore_index=True).to_csv(out / "mixed_model_omnibus.csv", index=False)
    pd.concat(local, ignore_index=True).to_csv(out / "mixed_model_age_localization.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(out / "mixed_model_optimizer_log.csv", index=False)
    pd.concat(robust_tests, ignore_index=True).to_csv(out / "fpn_robust_omnibus.csv", index=False)
    pd.concat(robust_local, ignore_index=True).to_csv(out / "fpn_robust_age_localization.csv", index=False)


if __name__ == "__main__":
    main()
