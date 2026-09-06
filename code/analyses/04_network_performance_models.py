#!/usr/bin/env python3
"""Bivariate and simultaneous three-network associations with task performance."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import variance_inflation_factor

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from lib.analysis_utils import (  # noqa: E402
    CARIT_DVS, FACENAME_DVS, NETWORK_COLUMNS, bh_fdr, coefficient_table, ensure_output_dir,
    fit_clustered_ols, fit_gee, joint_wald, read_analysis_table, require_columns, zscore
)


def bivariate(df: pd.DataFrame, task: str, dvs: dict[str, str], networks: dict[str, str]) -> pd.DataFrame:
    rows = []
    for dv_label, dv_col in dvs.items():
        for network, net_col in networks.items():
            x = pd.to_numeric(df[net_col], errors="coerce")
            y = pd.to_numeric(df[dv_col], errors="coerce")
            ok = x.notna() & y.notna()
            r, p = pearsonr(x[ok], y[ok])
            rows.append({"task": task, "outcome": dv_label, "network": network, "N": ok.sum(), "r": r, "p": p})
    out = pd.DataFrame(rows)
    out["q_BH"] = bh_fdr(out["p"])
    return out


def vif_table(df: pd.DataFrame, task: str, networks: dict[str, str]) -> pd.DataFrame:
    cols = list(networks.values())
    d = df[cols].apply(pd.to_numeric, errors="coerce").dropna()
    X = np.column_stack([np.ones(len(d)), d.to_numpy()])
    rows = []
    for i, (network, col) in enumerate(networks.items(), start=1):
        rows.append({"task": task, "network": network, "column": col, "N": len(d), "VIF": variance_inflation_factor(X, i)})
    return pd.DataFrame(rows)


def simultaneous_models(df: pd.DataFrame, task: str, dvs: dict[str, str], networks: dict[str, str]):
    predictor_cols = list(networks.values())
    rows_models, rows_coef = [], []
    for dv_label, dv_col in dvs.items():
        use = ["SubjectID", dv_col, *predictor_cols]
        d = df[use].dropna().copy()
        formula = f"Q('{dv_col}') ~ " + " + ".join([f"Q('{c}')" for c in predictor_cols])

        ols = smf.ols(formula, data=d).fit()
        gee = fit_gee(formula, d)
        cols_cluster = fit_clustered_ols(formula, d)

        for estimator, fit in [("Pooled OLS", ols), ("GEE robust", gee), ("Clustered OLS", cols_cluster)]:
            terms = [t for t in fit.params.index if t != "Intercept"]
            if estimator == "Pooled OLS":
                stat = {"chi2": np.nan, "df": len(terms), "p": fit.f_pvalue}
            else:
                stat = joint_wald(fit, terms)
            rows_models.append({
                "task": task, "outcome": dv_label, "estimator": estimator, "N": len(d),
                "R2": getattr(fit, "rsquared", np.nan), "omnibus_stat": stat["chi2"], "df": stat["df"], "p": stat["p"],
            })
            c = coefficient_table(fit, f"{task}_{dv_label}_{estimator}")
            c["task"] = task; c["outcome"] = dv_label; c["estimator"] = estimator
            rows_coef.append(c)
    coef = pd.concat(rows_coef, ignore_index=True)
    mask = coef.estimator.eq("GEE robust") & coef.term.ne("Intercept")
    coef.loc[mask, "q_BH_within_task"] = bh_fdr(coef.loc[mask, "p"])
    return pd.DataFrame(rows_models), coef


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", default="outputs/network_performance")
    args = parser.parse_args()

    df = read_analysis_table(args.input)
    out = ensure_output_dir(args.output_dir)
    bivar, vifs, models, coefs = [], [], [], []
    for task, dvs in [("CARIT", CARIT_DVS), ("FACENAME", FACENAME_DVS)]:
        networks = NETWORK_COLUMNS[task]
        require_columns(df, [*dvs.values(), *networks.values()], "analysis table")
        bivar.append(bivariate(df, task, dvs, networks))
        vifs.append(vif_table(df, task, networks))
        m, c = simultaneous_models(df, task, dvs, networks)
        models.append(m); coefs.append(c)
    pd.concat(bivar, ignore_index=True).to_csv(out / "bivariate_network_performance.csv", index=False)
    pd.concat(vifs, ignore_index=True).to_csv(out / "network_vif.csv", index=False)
    pd.concat(models, ignore_index=True).to_csv(out / "three_network_models.csv", index=False)
    pd.concat(coefs, ignore_index=True).to_csv(out / "three_network_coefficients.csv", index=False)


if __name__ == "__main__":
    main()
