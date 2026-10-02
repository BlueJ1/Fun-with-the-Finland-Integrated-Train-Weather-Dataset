# Replication evidence

The checked-in evidence contains our protocol/source/data audits, public metadata, the historical dependency export, and the original cohort manifests. Full upstream code snapshots and extracted paper text are local caches excluded from version control. Original native data and superseded diagnostic outputs are also excluded.

The source audits cite these pinned files by filename and line number:

- [src/training_pipeline.py](https://github.com/borinvini/Railway-FMI-Data_training/blob/c28b188948fe42ccc0da2c0f84a81405cc1e2a72/src/training_pipeline.py)
- [src/preprocessing_pipeline.py](https://github.com/borinvini/Railway-FMI-Data_training/blob/c28b188948fe42ccc0da2c0f84a81405cc1e2a72/src/preprocessing_pipeline.py)
- [config/const_training.py](https://github.com/borinvini/Railway-FMI-Data_training/blob/c28b188948fe42ccc0da2c0f84a81405cc1e2a72/config/const_training.py)
- [config/const_preprocessing.py](https://github.com/borinvini/Railway-FMI-Data_training/blob/c28b188948fe42ccc0da2c0f84a81405cc1e2a72/config/const_preprocessing.py)
- [environment.yml](https://github.com/borinvini/Railway-FMI-Data_training/blob/c28b188948fe42ccc0da2c0f84a81405cc1e2a72/environment.yml)
- [src/objective/regression_obj.cu](https://github.com/dmlc/xgboost/blob/v3.0.1/src/objective/regression_obj.cu)

[source_snapshot_manifest.json](source_snapshot_manifest.json) records SHA-256 hashes of the inspected snapshots. [cohort_manifest.json](cohort_manifest.json) records all 84 native input hashes, filtering, duplicate removal, feature rules, and the derived table hash. [legacy_cohort_manifest.json](legacy_cohort_manifest.json) records the approximate historical row-order policy and its table hash.

The source paper is [Predicting Train Delays in Finland Using Machine Learning and Weather Data](https://arxiv.org/abs/2609.11277). The [FI-TW dataset](https://www.kaggle.com/datasets/viniborin/finland-integrated-train-weather-dataset-fi-tw) metadata identifies GPL-3.0 licensing. Test prediction and split-membership files under results derive from this public dataset.
