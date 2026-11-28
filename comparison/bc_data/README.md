# Data for comparison/bc_scalars.py

* `clinical.json`: per BC-TCGA tumour sample id, `[PAM50, ER, OS event, OS months]`.
  PAM50 (`SUBTYPE`), OS status and months: cBioPortal study `brca_tcga_pan_can_atlas_2018`,
  `data_clinical_patient.txt` (TCGA PanCanAtlas). ER (`ER_STATUS_BY_IHC`, `+`/`-`): cBioPortal
  study `brca_tcga`, `data_clinical_patient.txt`. Matched by patient (first 12 characters of the
  barcode). Empty or null: not available. Taken from github.com/cbioportal/datahub, November 2026.
* `hallmark.json`: MSigDB Hallmark gene sets (v6.2, gene symbols) G2M checkpoint, E2F targets,
  estrogen response early and late, hypoxia (Liberzon et al., Cell Systems 2015).
