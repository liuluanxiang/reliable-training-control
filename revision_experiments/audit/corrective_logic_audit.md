# Corrective Logic Audit

## Fixed defects

1. **Stochastic validation observations:** CIFAR validation previously reused the augmented training dataset object. Protocol v2 uses a second dataset object with deterministic evaluation transforms over the same seed-fixed held-out indices.
2. **Unreachable reliability early stopping:** the LR patience reset prevented the reliability counter from reaching stop patience. Protocol v2 uses independent `bad_rel_lr` and `bad_rel_stop` counters.
3. **Redundant gradient-norm work:** the loop computed a full gradient norm for every minibatch but retained only the final value. Protocol v2 computes exactly the retained final-minibatch norm once per epoch.
4. **Invalid CUDA overhead attribution:** parameter-level `.item()` calls synchronized pending training kernels and counted that wait as controller work. Norm reductions now stay on-device, synchronize once, and use explicit timing boundaries; the initial PB reference is retained on-device to avoid repeated host-to-device copies.
5. **Unsafe completed-result reuse:** a `summary.json` at the expected path was previously treated as complete without checking scientific configuration or the remaining output files. Reuse now requires all nine artifacts and a matching scientific-run hash; partial outputs or mismatches are archived and rerun.
6. **Broken seed grouping:** scheduled single-run configs previously changed `experiment_name` for every method/seed, preventing five-seed statistical grouping. Temporary config filenames remain unique while the scientific experiment name remains stable.
7. **Missing fourth baseline in statistics:** Validation-Loss-Only is now included in the baseline set and paired comparisons.
8. **Missing revision postprocessing:** `analyze_revision.py` now materializes documented main-run sharing, summaries, paired bootstrap CIs, paired Cohen's d_z, raw tests, and Holm-corrected tests.
9. **Unrecorded queue failures:** `run_manifest.csv` is updated before and after every run; failures are retained and the queue can continue unless `--fail-fast` is requested.

## Verified invariants

- Training and internal validation indices are disjoint and reproducible by seed.
- Test evaluation occurs only after validation checkpoint selection.
- All five compared methods share data, model initialization, optimizer recipe, epoch ceiling, and checkpoint rule within a setting.
- Tiny ImageNet's labeled official validation split is isolated as final test data; its unlabeled official test split is unused.
- DeiT loads the official ImageNet-1k backbone while discarding only the 1,000-class head.
- Headline paired statistics are labeled headline-only at exactly five matched seeds.

No additional known core training/control logic gap remains after the regression suite and smoke tests. Long-run scientific conclusions remain pending the 260 corrected-protocol physical runs.
