# Data requirements

The repository does not include AABC/HCP participant-level data. The analysis scripts require a locally obtained participant-visit table assembled from AABC Release 2 under the applicable data-use terms.

## Cohort structure

The manuscript analysis uses a locked longitudinal cohort of 80 participants with four visits each (320 participant-visits). The public code validates this structure but does not embed participant identifiers. FACENAME-specific analyses use available observations within that cohort.

## Required identifiers and longitudinal covariates

- `SubjectID`
- `Visit` (`V1`–`V4`)
- `baseline_age_v1`
- `baseline_age_c`
- `time_years`
- `sex`
- `RelativeRMS_mean`
- `MR_QC_Issue_Codes`
- `qc_any`

If `baseline_age_v1`, `baseline_age_c`, or `time_years` are not populated, the preprocessing script can derive them from `age_open` and `days_from_V1`.

## CARIT behavioral inputs

For reconstruction of the six reported CARIT outcomes, the preprocessing script expects:

- `n_goHit`, `n_goMiss`, `n_goTotal`
- `n_nogoFA`, `n_nogoCR`, `n_nogoTotal`

It derives `HR`, `FAR`, `d_prime`, `criterion_c`, `CommissionErrorRate`, and `OmissionErrorRate`. HR and FAR boundary values of 0 or 1 are corrected to `0.5/N` and `(N - 0.5)/N` before probit transformation.

## FACENAME behavioral outcomes

The analysis scripts use the post-scan recall variables:

- `FACENAME_BEHAV__pct_correct`
- `FACENAME_BEHAV__pct_omission`
- `FACENAME_BEHAV__pct_incorrect`
- `FACENAME_BEHAV__conditional_accuracy_pct`

## Network Full-amplitude variables

CARIT:

- `DMN_Full_mean`
- `SN_Full_mean`
- `FPN_Full_mean`

FACENAME:

- `FACENAME_IDP_MEAN__DMN_Full_amplitude_mean`
- `FACENAME_IDP_MEAN__SN_Full_amplitude_mean`
- `FACENAME_IDP_MEAN__FPN_Full_amplitude_mean`

When parcel-level Full-amplitude columns are present with the prefixes below, the preprocessing script recomputes the corresponding network mean as an unweighted parcel average with no imputation:

- `CARIT_IDP_FULL__DMN:` — 77 parcels
- `CARIT_IDP_FULL__SN:` — 56 parcels
- `CARIT_IDP_FULL__FPN:` — 50 parcels
- `FACENAME_IDP_FULL__DMN:` — 77 parcels
- `FACENAME_IDP_FULL__SN:` — 56 parcels
- `FACENAME_IDP_FULL__FPN:` — 50 parcels, if supplied

The SN definition is the 56-parcel CAB-NP cingulo-opercular network used as the study’s operational salience-network analogue. The FPN uses the 50 cortical CAB-NP frontoparietal parcels. CARIT and FACENAME imaging are never substituted across tasks.

## Data that should not be committed

Do not commit participant-level AABC/HCP spreadsheets, CSVs, raw task files, amplitude ZIPs, subject identifiers extracted from those files, or generated analysis outputs. The repository `.gitignore` excludes common data/output formats by default.
