# PD Parameter Audit

- `lambda_g = 0.0001`.
- `lambda_u = 0.001`.
- Defined in every current JSON config and as defaults in `src/signals.py`; fixed across datasets/models.
- Whether systematically tuned and why these scales were selected: UNRESOLVED.
- Explicit variants added without changing the default: FULL, V only, V+H, V+U, and H+U.
