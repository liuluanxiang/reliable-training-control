# Data Leakage Protocol

Training data is used for optimization only. Validation data drives PB/PD observations, LR control, stopping, and maximum-validation-accuracy checkpoint selection. Test data is evaluated exactly once after the checkpoint is selected. New runs persist `test_evaluation_count`, `test_evaluation_timestamp`, `checkpoint_selected_before_test`, and `test_used_during_training`.

Automatic exclusion rule: any run with `test_used_during_training=true`, a test count other than one, or checkpoint selection after the test timestamp is invalid. Historical code follows the intended call order but did not persist these fields, so this is verified from code provenance rather than run metadata.

Protocol v2 correction: training and validation use separate dataset objects over disjoint seed-fixed indices. Validation uses only deterministic tensor conversion/normalization for native-size CNN runs and deterministic resize/center-crop for DeiT. Historical v1 runs are excluded from revision inference. Tiny ImageNet uses 90,000 official training images for optimization, 10,000 held-out official training images for controller/checkpoint decisions, and the 10,000 labeled official validation images for exactly one final test evaluation.
