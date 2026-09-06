#!/usr/bin/env python3
"""Sensitivity analyses reported for network-performance and longitudinal models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from lib.analysis_utils import (  # noqa: E402
    CARIT_DVS, FACENAME_DVS, NETWORK_COLUMNS, bh_fdr, ensure_output_dir, fit_clustered_ols, fit_gee,
    joint_wald, matching_terms, read_analysis_table, stack_outcomes, stack_networks
)


def merge_network(long: pd.DataFrame, df: pd.DataFrame, col: str) -> pd.DataFrame:
    return long.merge(df[["SubjectID", "Visit", col]], on=["SubjectID", "Visit"], how="left").dropna(subset=[col])


def fpn_age_test(df: pd.DataFrame, dvs: dict[str, str], fpn_col: str, extra: str = "") -> dict:
    long = merge_network(stack_outcomes(df, dvs), df, fpn_col)
    formula = f"z ~ C(DV_type) * Q('{fpn_col}') * baseline_age_c + time_years {extra}"
    fit = fit_gee(formula, long)
    terms = matching_terms(fit, ["C(DV_type)", f"Q('{fpn_col}')", "baseline_age_c"])
    stat = joint_wald(fit, terms)
    return {"chi2": stat["chi2"], "df": stat["df"], "p": stat["p"], "N_rows": len(long), "N_participants": long.SubjectID.nunique()}


def residualized_fpn(df: pd.DataFrame, task: str) -> pd.DataFrame:
    m = NETWORK_COLUMNS[task]
    use = ["SubjectID", "Visit", m["DMN"], m["SN"], m["FPN"]]
    d = df[use].dropna().copy()
    fit = smf.ols(f"Q('{m['FPN']}') ~ Q('{m['DMN']}') + Q('{m['SN']}')", data=d).fit()
    d["FPN_residual"] = fit.resid
    return d[["SubjectID", "Visit", "FPN_residual"]]


def residualized_models(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for task, dvs in [("CARIT", CARIT_DVS), ("FACENAME", FACENAME_DVS)]:
        r = residualized_fpn(df, task)
        x = df.merge(r, on=["SubjectID", "Visit"], how="left")
        stat = fpn_age_test(x, dvs, "FPN_residual")
        rows.append({"task": task, "sensitivity": "FPN residualized from DMN + SN", **stat})
    return pd.DataFrame(rows)


def carit_fpn_sensitivities(df: pd.DataFrame) -> pd.DataFrame:
    fpn = NETWORK_COLUMNS["CARIT"]["FPN"]
    rows = []
    rows.append({"sensitivity": "baseline", **fpn_age_test(df, CARIT_DVS, fpn)})

    if "RelativeRMS_mean" in df:
        rows.append({"sensitivity": "motion adjusted", **fpn_age_test(df, CARIT_DVS, fpn, "+ C(DV_type):RelativeRMS_mean")})
    if "sex" in df:
        rows.append({"sensitivity": "sex adjusted", **fpn_age_test(df, CARIT_DVS, fpn, "+ C(DV_type):C(sex)")})
    if "qc_any" in df:
        clean = df.loc[pd.to_numeric(df["qc_any"], errors="coerce").eq(0)].copy()
        rows.append({"sensitivity": "QC-clean visits", **fpn_age_test(clean, CARIT_DVS, fpn)})

    x = pd.to_numeric(df[fpn], errors="coerce")
    q1, q3 = x.quantile([0.25, 0.75]); iqr = q3 - q1
    clean = df.loc[~((x < q1 - 3 * iqr) | (x > q3 + 3 * iqr))].copy()
    rows.append({"sensitivity": ">3xIQR FPN excluded", **fpn_age_test(clean, CARIT_DVS, fpn)})

    for omit_label in ["Commission Error Rate", "False Alarm Rate"]:
        reduced = {k: v for k, v in CARIT_DVS.items() if k != omit_label}
        rows.append({"sensitivity": f"omit {omit_label}", **fpn_age_test(df, reduced, fpn)})
    return pd.DataFrame(rows)


def motion_adjusted_network_age(df: pd.DataFrame) -> pd.DataFrame:
    if "RelativeRMS_mean" not in df:
        return pd.DataFrame()
    rows = []
    for task, mapping in NETWORK_COLUMNS.items():
        long = stack_networks(df, mapping).dropna(subset=["RelativeRMS_mean"])
        formula = "z_amplitude ~ 0 + C(Network) + C(Network):baseline_age_c + C(Network):time_years + C(Network):RelativeRMS_mean"
        fit = fit_gee(formula, long)
        for effect, token in [("baseline_age", "baseline_age_c"), ("longitudinal_time", "time_years")]:
            terms = matching_terms(fit, ["C(Network)", token])
            stat = joint_wald(fit, terms)
            rows.append({"task": task, "effect": effect, **{k: stat[k] for k in ["chi2", "df", "p"]}})
    return pd.DataFrame(rows)


def facename_categorical_visit(df: pd.DataFrame):
    long = stack_outcomes(df, FACENAME_DVS)
    long["Visit"] = pd.Categorical(long["Visit"], categories=["V1", "V2", "V3", "V4"], ordered=True)
    # V1 is the reference. The 12-df joint test combines the three Visit main
    # effects with the nine DV-specific Visit interactions.
    formula = "z ~ C(DV_type) * C(Visit) + C(DV_type):baseline_age_c"
    gee = fit_gee(formula, long)
    ols = fit_clustered_ols(formula, long)
    rows = []
    for estimator, fit in [("GEE robust", gee), ("Clustered OLS", ols)]:
        terms = [name for name in fit.params.index if "C(Visit)" in name]
        stat = joint_wald(fit, terms)
        rows.append({"estimator": estimator, "test": "overall outcome-specific visit profile", **{k: stat[k] for k in ["chi2", "df", "p"]}})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/sensitivity")
    args = parser.parse_args()
    df = read_analysis_table(args.input)
    out = ensure_output_dir(args.output_dir)

    residualized_models(df).to_csv(out / "residualized_fpn.csv", index=False)
    carit_fpn_sensitivities(df).to_csv(out / "carit_fpn_sensitivities.csv", index=False)
    motion_adjusted_network_age(df).to_csv(out / "motion_adjusted_network_age.csv", index=False)
    facename_categorical_visit(df).to_csv(out / "facename_categorical_visit.csv", index=False)


if __name__ == "__main__":
    main()
