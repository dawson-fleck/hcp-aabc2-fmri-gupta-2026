# Script-to-manuscript mapping

| Script | Manuscript content reproduced |
|---|---|
| `00_prepare_analysis_data.py` | CARIT behavioral derivation; baseline-age/time variables; QC flags; network Full-amplitude aggregation |
| `01_descriptive_qc.py` | Participant/task/network descriptive summaries; motion and imaging QC; network intercorrelations; IQR review flags |
| `02_behavior_age_longitudinal.py` | Direct baseline-age and longitudinal-time associations with CARIT and FACENAME performance; per-outcome estimates and FDR correction |
| `03_network_age_models.py` | Baseline-age and longitudinal-time associations with DMN, SN, and FPN Full amplitude |
| `04_network_performance_models.py` | Pooled bivariate screens; simultaneous three-network OLS/GEE models; clustered OLS; VIF diagnostics |
| `05_age_moderation_models.py` | Repeated-measures network-performance models; DMN/SN/FPN baseline-age moderation; robust FPN estimator checks |
| `06_sensitivity_analyses.py` | Residualized FPN; motion/QC/outlier/sex sensitivity; FAR/commission leave-one-out checks; FACENAME categorical-visit sensitivity |
| `07_visit1_age_group_tests.py` | Visit-1 Younger-versus-Older DMN/SN/FPN comparisons with Welch tests, Hedges’ g, and Holm correction |
| `08_make_statistical_figures.py` | Statistical components of the longitudinal behavioral and age-dependent FPN figures |

The scripts intentionally omit exploratory analyses that were not retained in the manuscript. The repository is designed as a reproducibility package for the reported paper, not as an archive of every exploratory analysis performed during project development.
