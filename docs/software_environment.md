# Software environment

The recorded analysis environment used Python 3.14.6 with:

- pandas 3.0.5
- NumPy 2.5.1
- statsmodels 0.14.6

The exact SciPy patch version was not retained in the original environment record, so the repository specifies a compatible minimum version rather than inventing an exact value. Matplotlib and openpyxl are included because the public scripts generate figures and read Excel source files.

For the neuroimaging products analyzed in the manuscript, processing provenance indicated HCP Pipelines 4.7.0 within QuNex 0.96.2, with FreeSurfer 6.0.0, FSL 6.0.5.1, and Connectome Workbench 1.5.0. Those packages were used upstream to generate released AABC/HCP imaging products; this repository does not rerun image-level preprocessing.
