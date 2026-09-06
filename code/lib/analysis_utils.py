"""Shared utilities for the AABC task-fMRI aging analyses."""

from __future__ import annotations

import math
import re
import warnings
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import chi2, norm
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests
from statsmodels.tools.sm_exceptions import ConvergenceWarning

CARIT_DVS = {
    "Hit Rate": "HR",
    "False Alarm Rate": "FAR",
    "d_prime": "d_prime",
    "criterion c": "criterion_c",
    "Commission Error Rate": "CommissionErrorRate",
    "Omission Error Rate": "OmissionErrorRate",
}

FACENAME_DVS = {
    "Percent Correct": "FACENAME_BEHAV__pct_correct",
    "Percent Omission": "FACENAME_BEHAV__pct_omission",
    "Percent Incorrect/Intrusion": "FACENAME_BEHAV__pct_incorrect",
    "Conditional Accuracy": "FACENAME_BEHAV__conditional_accuracy_pct",
}

NETWORK_COLUMNS = {
    "CARIT": {
        "DMN": "DMN_Full_mean",
        "SN": "SN_Full_mean",
        "FPN": "FPN_Full_mean",
    },
    "FACENAME": {
        "DMN": "FACENAME_IDP_MEAN__DMN_Full_amplitude_mean",
        "SN": "FACENAME_IDP_MEAN__SN_Full_amplitude_mean",
        "FPN": "FACENAME_IDP_MEAN__FPN_Full_amplitude_mean",
    },
}

OPTIMIZERS = ["lbfgs", "powell", "bfgs", "nm", "cg"]


def read_analysis_table(path: str | Path, sheet_name: str = "CARIT_master_combined") -> pd.DataFrame:
    """Read the participant-visit table used by the analysis scripts.

    The source workbook uses a documentation row followed by the true header row.
    A CSV input is also accepted for users who export the analysis table first.
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet_name, header=1)
    if "SubjectID" not in df or "Visit" not in df:
        raise KeyError("Input must contain SubjectID and Visit columns.")
    df = df.copy()
    df["SubjectID"] = df["SubjectID"].astype(str).str.strip()
    df["Visit"] = df["Visit"].astype(str).str.strip()
    return df


def require_columns(df: pd.DataFrame, columns: Iterable[str], context: str = "input") -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(f"{context} is missing required columns: {missing}")


def validate_locked_cohort(df: pd.DataFrame, expected_subjects: int = 80, expected_visits: int = 4) -> None:
    """Validate the locked longitudinal cohort without embedding subject identifiers."""
    n_subjects = df["SubjectID"].nunique()
    if n_subjects != expected_subjects:
        raise ValueError(f"Expected {expected_subjects} participants; found {n_subjects}.")
    visit_counts = df.groupby("SubjectID")["Visit"].nunique()
    if not (visit_counts == expected_visits).all():
        bad = visit_counts[visit_counts != expected_visits]
        raise ValueError("Each participant must have four visits.\n" + bad.to_string())
    if df.duplicated(["SubjectID", "Visit"]).any():
        raise ValueError("Duplicate SubjectID + Visit rows detected.")


def zscore(series: pd.Series) -> pd.Series:
    x = pd.to_numeric(series, errors="coerce")
    sd = x.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.nan, index=x.index, dtype=float)
    return (x - x.mean()) / sd


def stack_outcomes(df: pd.DataFrame, dv_map: Mapping[str, str]) -> pd.DataFrame:
    """Standardize each outcome, then stack outcomes into long format."""
    base = ["SubjectID", "Visit", "baseline_age_c", "time_years"]
    optional = [c for c in ["sex", "RelativeRMS_mean", "qc_any"] if c in df.columns]
    rows = []
    for label, col in dv_map.items():
        require_columns(df, [col, *base], "analysis table")
        x = df[base + optional + [col]].copy()
        x["DV_type"] = label
        x["z"] = zscore(x[col])
        rows.append(x.drop(columns=[col]))
    out = pd.concat(rows, ignore_index=True)
    out["DV_type"] = pd.Categorical(out["DV_type"], categories=list(dv_map.keys()), ordered=True)
    return out.dropna(subset=["z", "baseline_age_c", "time_years"])


def stack_networks(df: pd.DataFrame, network_map: Mapping[str, str]) -> pd.DataFrame:
    """Standardize each network amplitude, then stack networks into long format."""
    base = ["SubjectID", "Visit", "baseline_age_c", "time_years"]
    optional = [c for c in ["sex", "RelativeRMS_mean", "qc_any"] if c in df.columns]
    rows = []
    for label, col in network_map.items():
        require_columns(df, [col, *base], "analysis table")
        x = df[base + optional + [col]].copy()
        x["Network"] = label
        x["z_amplitude"] = zscore(x[col])
        rows.append(x.drop(columns=[col]))
    out = pd.concat(rows, ignore_index=True)
    out["Network"] = pd.Categorical(out["Network"], categories=list(network_map.keys()), ordered=True)
    return out.dropna(subset=["z_amplitude", "baseline_age_c", "time_years"])


def bh_fdr(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out


def holm(p_values: Iterable[float]) -> np.ndarray:
    p = np.asarray(list(p_values), dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        out[ok] = multipletests(p[ok], method="holm")[1]
    return out


def fit_gee(formula: str, data: pd.DataFrame, cov_type: str = "robust"):
    """Gaussian GEE with exchangeable within-participant working correlation."""
    model = smf.gee(
        formula,
        groups="SubjectID",
        data=data,
        family=sm.families.Gaussian(),
        cov_struct=sm.cov_struct.Exchangeable(),
    )
    return model.fit(cov_type=cov_type)


def fit_clustered_ols(formula: str, data: pd.DataFrame):
    """OLS with participant-clustered sandwich covariance."""
    model = smf.ols(formula, data=data)
    return model.fit(cov_type="cluster", cov_kwds={"groups": data["SubjectID"]})


def matching_terms(result, include: Iterable[str], exclude: Iterable[str] = ()) -> list[str]:
    names = list(result.params.index)
    include = list(include)
    exclude = list(exclude)
    return [n for n in names if all(s in n for s in include) and not any(s in n for s in exclude)]


def joint_wald(result, terms: Iterable[str]) -> dict:
    """Joint Wald chi-square test for a set of model coefficients."""
    terms = list(terms)
    if not terms:
        raise ValueError("No model terms supplied to joint_wald.")
    names = list(result.params.index)
    idx = [names.index(t) for t in terms]
    beta = np.asarray(result.params, dtype=float)[idx]
    cov = np.asarray(result.cov_params(), dtype=float)[np.ix_(idx, idx)]
    inv = np.linalg.pinv(cov)
    stat = float(beta.T @ inv @ beta)
    df = int(np.linalg.matrix_rank(cov))
    return {"chi2": stat, "df": df, "p": float(chi2.sf(stat, df)), "terms": "; ".join(terms)}


def coefficient_table(result, model_label: str) -> pd.DataFrame:
    ci = result.conf_int()
    return pd.DataFrame({
        "model": model_label,
        "term": result.params.index,
        "estimate": result.params.values,
        "SE": result.bse.values,
        "p": result.pvalues.values,
        "CI95_low": ci.iloc[:, 0].values,
        "CI95_high": ci.iloc[:, 1].values,
    })


def fit_best_mixedlm(formula: str, data: pd.DataFrame, reml: bool = False):
    """Fit a random-intercept mixed model with several optimizers.

    The highest-log-likelihood converged solution is retained, matching the
    optimizer strategy used in the manuscript analyses.
    """
    candidates = []
    log_rows = []
    for method in OPTIMIZERS:
        try:
            model = smf.mixedlm(formula, data=data, groups=data["SubjectID"], re_formula="1")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                res = model.fit(reml=reml, method=method, maxiter=5000, disp=False)
            converged = bool(getattr(res, "converged", False))
            llf = float(res.llf)
            log_rows.append({
                "optimizer": method,
                "reml": reml,
                "converged": converged,
                "logLik": llf,
                "warnings": " | ".join(str(w.message) for w in caught),
            })
            if converged and np.isfinite(llf):
                candidates.append((llf, method, res))
        except Exception as exc:
            log_rows.append({
                "optimizer": method,
                "reml": reml,
                "converged": False,
                "logLik": np.nan,
                "warnings": f"{type(exc).__name__}: {exc}",
            })
    if not candidates:
        raise RuntimeError(f"No mixed-model optimizer converged for: {formula}")
    candidates.sort(key=lambda x: x[0], reverse=True)
    llf, method, res = candidates[0]
    return res, pd.DataFrame(log_rows), method


def likelihood_ratio_test(full, reduced) -> dict:
    stat = max(0.0, 2.0 * (float(full.llf) - float(reduced.llf)))
    df = int(len(full.fe_params) - len(reduced.fe_params))
    if df <= 0:
        raise ValueError("Reduced model does not have fewer fixed-effect parameters.")
    return {"chi2": stat, "df": df, "p": float(chi2.sf(stat, df))}


def linear_combination(result, terms: Iterable[str]) -> dict:
    """Estimate a linear combination of fixed-effect coefficients."""
    terms = list(terms)
    params = result.fe_params if hasattr(result, "fe_params") else result.params
    names = list(params.index)
    v = np.zeros(len(names), dtype=float)
    for term in terms:
        if term not in names:
            raise KeyError(f"Term not found: {term}")
        v[names.index(term)] += 1.0
    cov = result.cov_params().loc[names, names].to_numpy(dtype=float)
    beta = params.to_numpy(dtype=float)
    est = float(v @ beta)
    se = math.sqrt(max(float(v @ cov @ v), 0.0))
    z = est / se if se else np.nan
    p = float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
    return {
        "estimate": est,
        "SE": se,
        "z": z,
        "p": p,
        "CI95_low": est - 1.959963984540054 * se,
        "CI95_high": est + 1.959963984540054 * se,
    }


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def ensure_output_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
