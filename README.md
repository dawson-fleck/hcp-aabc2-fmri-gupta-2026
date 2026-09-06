# Task-fMRI network correlates of cognitive aging

This repository contains the custom code used to prepare and analyze the participant-visit data for the manuscript **“Dissociable Network Correlates of Inhibitory Control and Associative Memory Across Aging: A Study on Default Mode, Salience and Frontoparietal Networks.”**

The repository contains code and documentation only. AABC/HCP imaging, behavioral, phenotypic, and participant-level derived data are **not** redistributed here. Researchers must obtain the required source data separately under the applicable AABC/HCP access terms.

## Repository structure

```text
code/
  preprocessing/
    00_prepare_analysis_data.py
  analyses/
    01_descriptive_qc.py
    02_behavior_age_longitudinal.py
    03_network_age_models.py
    04_network_performance_models.py
    05_age_moderation_models.py
    06_sensitivity_analyses.py
    07_visit1_age_group_tests.py
  figures/
    08_make_statistical_figures.py
  lib/
    analysis_utils.py
docs/
  analysis_workflow.md
  data_requirements.md
  manuscript_mapping.md
  software_environment.md
  surface_map_workflow.md
requirements.txt
.gitignore
LICENSE
```

## Analysis scope

The code reproduces the analyses reported in the manuscript: CARIT inhibitory-control performance, FACENAME associative-memory performance, task-specific Default Mode Network (DMN), Salience Network (SN; operationalized from the CAB-NP cingulo-opercular network), and Frontoparietal Network (FPN) Full-amplitude measures, longitudinal and baseline-age models, network-performance associations, baseline-age moderation, and reported QC/sensitivity analyses.

Abandoned exploratory analyses and analyses not reported in this manuscript are intentionally not included.

## Quick start

Create an environment and install the dependencies:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Prepare the analysis table from a locally obtained AABC-derived participant-visit workbook:

```bash
python code/preprocessing/00_prepare_analysis_data.py \
  --input /path/to/AABC2_analysis_source.xlsx \
  --output data/analysis_data.csv
```

Run the reported analyses:

```bash
python code/analyses/01_descriptive_qc.py --input data/analysis_data.csv
python code/analyses/02_behavior_age_longitudinal.py --input data/analysis_data.csv
python code/analyses/03_network_age_models.py --input data/analysis_data.csv
python code/analyses/04_network_performance_models.py --input data/analysis_data.csv
python code/analyses/05_age_moderation_models.py --input data/analysis_data.csv
python code/analyses/06_sensitivity_analyses.py --input data/analysis_data.csv
python code/analyses/07_visit1_age_group_tests.py --input data/analysis_data.csv
```

Generated data and result files are written under `outputs/` and are ignored by Git. See `docs/analysis_workflow.md` for the analysis sequence and `docs/manuscript_mapping.md` for the correspondence between scripts and manuscript sections.

## Reproducibility notes

The locked longitudinal analysis cohort contains 80 participants with four visits each. The public scripts validate that structure but do not embed participant identifiers. Behavioral signal-detection measures are reconstructed from CARIT trial counts with the same boundary correction used in the manuscript. Network Full-amplitude means are recomputed from tagged parcel columns when those parcel-level values are present in the local source table. No imputation or substitution between CARIT and FACENAME imaging is performed.

The statistical framework includes Gaussian generalized estimating equations with exchangeable within-participant working correlation and robust sandwich covariance, participant-clustered OLS sensitivity models, random-intercept mixed models with multiple optimizers for likelihood-ratio tests, Benjamini-Hochberg FDR control within prespecified families, and the reported motion/QC/outlier/sex/residualization sensitivity analyses.

## Data access

AABC Release 2 source data must be obtained separately from the Human Connectome Project/AABC distribution system. See `docs/data_requirements.md` for the fields expected by the scripts.
