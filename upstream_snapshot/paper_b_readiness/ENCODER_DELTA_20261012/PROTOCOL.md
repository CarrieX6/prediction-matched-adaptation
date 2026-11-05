# Paired encoder-effect comparison

BiomedCLIP and CLIP predictions are aligned by cohort, repetition, and case. Each bootstrap draw resamples the same patient clusters for both encoders. The reported encoder contrast is the CLIP Cflex-to-M gain minus the BiomedCLIP Cflex-to-M gain. CBIS-DDSM uses seed 20261021 and Derm7pt uses seed 20261022, with 10,000 draws per cohort.
