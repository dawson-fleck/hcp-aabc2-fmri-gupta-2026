# Cortical surface-map workflow

The manuscript’s cortical network and age-group maps are descriptive visualizations of parcel-level Full amplitude and are separate from the inferential network-mean models.

The surface-map workflow requires locally available AABC parcel-level Full-amplitude values, the HCP-MMP1.0 cortical parcellation, the CAB-NP parcel assignments used in the study, and Connectome Workbench. For Visit 1 age-group maps, parcel-level Full amplitude is averaged separately within the Younger (37–59 years) and Older (70–88 years) groups for each task and target network. These parcel means are mapped to the corresponding HCP-MMP1.0 parcels for visualization; parcels outside the target network are displayed as background cortex.

Statistical inference for the manuscript is performed on the network-mean Full-amplitude values, not on parcel-wise surface-map values. The public repository therefore does not include AABC participant-level parcel files or generated cortical maps.
