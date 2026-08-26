# Early-Stopping Logic

- Reliability improvement: `rbar < best_rbar - epsilon`.
- Validation improvement: `val_loss < best_val_loss - epsilon`.
- Reliability improvement resets both `bad_rel_lr` and `bad_rel_stop`; otherwise both increment.
- When `bad_rel_lr >= p_lr`, LR becomes `max(lr*gamma, min_lr)` and only `bad_rel_lr` resets.
- Stop condition: `bad_val >= p_stop OR bad_rel_stop >= p_stop`.
- This split-counter implementation makes reliability stopping reachable while retaining periodic LR reductions. `bad_rel` remains a history alias for `bad_rel_stop` for compatibility.
- Checkpoint retention: the epoch with strictly highest validation accuracy.

```text
update bad_rel_lr and bad_rel_stop using smoothed reliability score and epsilon
update bad_val using validation NLL and epsilon
if bad_rel_lr >= p_lr:
    reduce LR; bad_rel_lr = 0
if bad_val >= p_stop or bad_rel_stop >= p_stop:
    stop
retain model state when validation accuracy strictly improves
```
