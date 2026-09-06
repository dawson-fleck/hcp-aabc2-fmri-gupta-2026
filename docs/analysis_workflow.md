# Analysis workflow

The scripts are ordered by dependency rather than by manuscript section. They are intended to be run from the repository root.

1. **Prepare the participant-visit table** — `code/preprocessing/00_prepare_analysis_data.py` recalculates CARIT Hit Rate, False Alarm Rate, d′, criterion c, commission error rate, and omission error rate; constructs baseline-age and longitudinal-time variables where needed; derives QC flags; and recomputes network Full-amplitude means from tagged parcel columns when available.
2. **Descriptive and QC summaries** — `01_descriptive_qc.py` produces descriptive distributions, network intercorrelations, motion-network correlations, QC counts, and Tukey 1.5×/3× IQR review flags.
3. **Behavioral aging models** — `02_behavior_age_longitudinal.py` fits stacked Gaussian GEE models for omnibus baseline-age and longitudinal-time effects, plus per-outcome GEE and participant-clustered OLS models with within-family BH-FDR correction.
4. **Age and network amplitude** — `03_network_age_models.py` standardizes DMN/SN/FPN Full amplitude within network, stacks networks, and fits task-specific GEE models separating baseline age from elapsed follow-up.
5. **Age-independent network-performance models** — `04_network_performance_models.py` runs pooled Pearson screens, simultaneous DMN/SN/FPN OLS models, repeated-measures GEE versions, participant-clustered OLS sensitivity models, and VIF diagnostics.
6. **Network × baseline-age moderation** — `05_age_moderation_models.py` runs network-specific random-intercept mixed models and the FPN robust-estimator analyses. Likelihood-ratio tests use maximum-likelihood fits; outcome-specific localization uses REML. Multiple mixed-model optimizers are attempted and the highest-log-likelihood converged solution is retained.
7. **Sensitivity analyses** — `06_sensitivity_analyses.py` implements residualized-FPN, motion-adjusted, QC-clean, >3×IQR outlier, sex-adjusted, leave-one-out CARIT false-alarm/commission, motion-adjusted age-to-network, and FACENAME categorical-visit checks.
8. **Visit-1 age-group comparisons** — `07_visit1_age_group_tests.py` compares Younger (37–59) and Older (70–88) participants with Welch t-tests, Hedges’ g, and Holm correction across the six task × network comparisons.
9. **Statistical figures** — `08_make_statistical_figures.py` generates the longitudinal behavioral and FPN age-moderation statistical plots from the prepared data and model outputs. Cortical surface maps are documented separately.

All inferential models keep CARIT and FACENAME separate and use task-matched imaging variables. Full amplitude is the imaging endpoint used by these scripts; Partial amplitude is not part of the reported inferential pipeline.
