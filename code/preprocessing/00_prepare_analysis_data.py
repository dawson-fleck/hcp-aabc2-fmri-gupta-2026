#!/usr/bin/env python3
"""Construct the analysis-ready participant-visit table from AABC-derived inputs.

The script does not redistribute AABC/HCP data. It expects a locally obtained
participant-visit workbook containing the released/derived variables described
in docs/data_requirements.md and writes a compact analysis table used by the
remaining scripts.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

CODE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CODE_ROOT))
from lib.analysis_utils import (  # noqa: E402
    CARIT_DVS,
    FACENAME_DVS,
    NETWORK_COLUMNS,
    read_analysis_table,
    require_columns,
    validate_locked_cohort,
)

EXPECTED_PARCELS = {"DMN": 77, "SN": 56, "FPN": 50}


def edge_correct(rate: pd.Series, n: pd.Series) -> pd.Series:
    rate = pd.to_numeric(rate, errors="coerce").astype(float)
    n = pd.to_numeric(n, errors="coerce").astype(float)
    out = rate.copy()
    out = out.mask((rate == 0) & (n > 0), 0.5 / n)
    out = out.mask((rate == 1) & (n > 0), (n - 0.5) / n)
    return out


def derive_carit_behavior(df: pd.DataFrame) -> pd.DataFrame:
    required = ["n_goHit", "n_goMiss", "n_goTotal", "n_nogoFA", "n_nogoCR", "n_nogoTotal"]
    require_columns(df, required, "participant-visit table")
    out = df.copy()

    raw_hr = pd.to_numeric(out["n_goHit"], errors="coerce") / pd.to_numeric(out["n_goTotal"], errors="coerce")
    raw_far = pd.to_numeric(out["n_nogoFA"], errors="coerce") / pd.to_numeric(out["n_nogoTotal"], errors="coerce")
    out["HR"] = edge_correct(raw_hr, out["n_goTotal"])
    out["FAR"] = edge_correct(raw_far, out["n_nogoTotal"])
    out["CommissionErrorRate"] = raw_far
    out["OmissionErrorRate"] = pd.to_numeric(out["n_goMiss"], errors="coerce") / pd.to_numeric(out["n_goTotal"], errors="coerce")
    out["d_prime"] = norm.ppf(out["HR"]) - norm.ppf(out["FAR"])
    out["criterion_c"] = -0.5 * (norm.ppf(out["HR"]) + norm.ppf(out["FAR"]))
    return out


def derive_longitudinal_covariates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "baseline_age_v1" not in out or out["baseline_age_v1"].isna().all():
        require_columns(out, ["age_open"], "participant-visit table")
        v1 = out.loc[out["Visit"].eq("V1"), ["SubjectID", "age_open"]].drop_duplicates("SubjectID")
        v1 = v1.rename(columns={"age_open": "baseline_age_v1"})
        out = out.drop(columns=["baseline_age_v1"], errors="ignore").merge(v1, on="SubjectID", how="left")
    mean_age = pd.to_numeric(out["baseline_age_v1"], errors="coerce").drop_duplicates().mean()
    out["baseline_age_c"] = pd.to_numeric(out["baseline_age_v1"], errors="coerce") - mean_age

    if "time_years" not in out or out["time_years"].isna().all():
        require_columns(out, ["days_from_V1"], "participant-visit table")
        out["time_years"] = pd.to_numeric(out["days_from_V1"], errors="coerce") / 365.25
    return out


def derive_qc_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Preserve the released/project-verified QC derivative when it is already populated.
    if "qc_any" in out and pd.to_numeric(out["qc_any"], errors="coerce").notna().any():
        return out
    if "MR_QC_Issue_Codes" not in out:
        return out
    code = out["MR_QC_Issue_Codes"]
    text = code.astype("string").str.strip()
    missing = code.isna() | text.eq("") | text.str.lower().isin(["nan", "none", "missing", "unknown"])
    no_code = text.str.lower().isin(["0", "no code", "none reported", "no_issue", "no issue"])
    out["qc_any"] = np.where(missing, np.nan, np.where(no_code, 0, 1))
    out["qc_status"] = np.where(missing, "Missing/unknown", np.where(no_code, "No code", "Issue code(s) present"))
    return out


def derive_network_means_from_tagged_parcels(df: pd.DataFrame) -> pd.DataFrame:
    """Recompute Full-amplitude means from parcel columns when those columns are available."""
    out = df.copy()
    prefix_map = {
        ("CARIT", "DMN"): "CARIT_IDP_FULL__DMN:",
        ("CARIT", "SN"): "CARIT_IDP_FULL__SN:",
        ("CARIT", "FPN"): "CARIT_IDP_FULL__FPN:",
        ("FACENAME", "DMN"): "FACENAME_IDP_FULL__DMN:",
        ("FACENAME", "SN"): "FACENAME_IDP_FULL__SN:",
        ("FACENAME", "FPN"): "FACENAME_IDP_FULL__FPN:",
    }
    for (task, network), prefix in prefix_map.items():
        parcel_cols = [c for c in out.columns if str(c).startswith(prefix)]
        if not parcel_cols:
            continue
        expected = EXPECTED_PARCELS[network]
        if len(parcel_cols) != expected:
            raise ValueError(f"{task} {network}: expected {expected} tagged parcel columns, found {len(parcel_cols)}.")
        mean_col = NETWORK_COLUMNS[task][network]
        numeric = out[parcel_cols].apply(pd.to_numeric, errors="coerce")
        out[mean_col] = numeric.mean(axis=1, skipna=False)
    return out


def select_public_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    required = [
        "SubjectID", "Visit", "baseline_age_v1", "baseline_age_c", "time_years", "sex",
        "RelativeRMS_mean", "qc_any", "MR_QC_Issue_Codes",
        *CARIT_DVS.values(), *FACENAME_DVS.values(),
        *NETWORK_COLUMNS["CARIT"].values(), *NETWORK_COLUMNS["FACENAME"].values(),
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Prepared table is missing manuscript variables: {missing}")
    return df[required].copy()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="AABC-derived participant-visit workbook or CSV")
    parser.add_argument("--output", default="analysis_data.csv", help="Output CSV path")
    parser.add_argument("--sheet", default="CARIT_master_combined")
    parser.add_argument("--skip-cohort-check", action="store_true", help="Allow exploratory use outside the locked 80 x 4 cohort")
    args = parser.parse_args()

    df = read_analysis_table(args.input, args.sheet)
    if not args.skip_cohort_check:
        validate_locked_cohort(df)
    df = derive_carit_behavior(df)
    df = derive_longitudinal_covariates(df)
    df = derive_qc_flags(df)
    df = derive_network_means_from_tagged_parcels(df)
    out = select_public_analysis_columns(df)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Wrote {len(out)} participant-visits to {args.output}")


if __name__ == "__main__":
    main()
