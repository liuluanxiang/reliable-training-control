# Current Implementation Audit

Generated: 2026-08-26T00:31:49.304626+08:00

## Controller

- PB signal: `validation_NLL + sqrt((C_t + log(2*sqrt(n)/delta))/(2*n))`, where `C_t = ||theta_t-theta_0||^2/(2*sigma^2)`.
- PD signal: `max(0, validation_NLL-train_NLL) + lambda_g*gradient_norm + lambda_u*update_norm`; gradient norm is sampled on the final training minibatch and update norm is the full epoch parameter displacement.
- Both signals use online sample z-standardization (Welford update), then alpha weighting and beta exponential smoothing.
- LR is multiplied by gamma after `p_lr` non-improving reliability-score epochs. A separate non-resetting reliability-stagnation counter and the validation-stagnation counter independently trigger stopping at `p_stop`.
- Checkpoint selection is maximum validation accuracy; test evaluation occurs once after checkpoint selection.
- Optimizer/config: SGD, lr=0.1, momentum=0.9, weight decay=0.0005, batch=128; max epochs=120 for main runs and 90 for historical ablation/sensitivity.
- Seeds: 1..5 for ResNet-18 main; 1..3 historically for ResNet-50/VGG-16.
- Metrics: ECE uses 15 equal-width confidence bins; Brier is mean multiclass squared probability error; AURC integrates selective risk over coverage.

## Audit Outcome

- Historical main candidates reusable under corrected protocol: 0/150.
- Required corrected-protocol main rows: 150/150.
- All historical results are invalidated for revision inference: validation subsets inherited random crop/flip and the shared reliability counter made reliability-based stopping unreachable. Files remain untouched for provenance.
- New CIFAR and Tiny ImageNet validation subsets use separate dataset objects with deterministic evaluation transforms and seed-reproducible, disjoint indices.
- Baseline runs also compute diagnostic PB/PD trajectories; their recorded wall-clock therefore includes monitoring work.
- The legacy loop redundantly computed gradient norm on every minibatch while retaining only the final value. Protocol v2 computes it only on the final minibatch, preserving the signal definition and removing overwritten work.
- Tiny ImageNet support and a ResNet-18 recipe are present. The official training set supplies optimization/internal validation; the official labeled validation split is reserved for one final test evaluation.
- DeiT-Tiny/16 at 224 pixels is implemented with a local official ImageNet-1k checkpoint and a seed-specific CIFAR-100 classification head.
