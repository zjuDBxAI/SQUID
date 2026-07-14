# Veda Cost Model Calibration

This folder reproduces the Veda Appendix B calibration:

```text
C_theta(N, efs) = a * log2(N) + b * efs + c
```

The script uses vectors already stored in PostgreSQL `documentblocks`.

## Run

```bash
/home/chenyang/.conda/envs/multitenant/bin/python \
  /data/Multitenanthakes/controller/baseline/veda/train/train_cost_model.py \
  --sizes 5000,20000,80000 \
  --efs-values 1,5,10,20,40,80,120,200 \
  --query-limit 50 \
  --rebuild
```

Outputs are written to `controller/baseline/veda/train/result/`:

- `veda_cost_size_sweep.csv`
- `veda_cost_efs_sweep.csv`
- `veda_cost_fit.json`
- `veda_cost_fit.md`

## Method

1. Build deterministic sampled HNSW tables from current PostgreSQL data.
2. Size sweep: fix `efs=1`, vary `N`, and fit:

   ```text
   T_size(N) = a * log2(1 + N) + c1
   ```

3. Beam sweep: fix one `N`, vary `efs`, and fit both:

   ```text
   T_linear(efs) = b * efs + c2
   T_log(efs) = b' * efs * log2(efs) + c2'
   ```

4. Use the linear model when it has better R2, matching the Veda paper's
   intended cost model. The script still reports both fits.
5. Combine intercepts:

   ```text
   c_from_size_raw = c1 - b * 1
   c_from_efs_raw  = c2 - a * log2(1 + N0)

   c_from_size = max(c_from_size_raw, 0)
   c_from_efs  = max(c_from_efs_raw, 0)
   c = (c_from_size + c_from_efs) / 2
   ```

   The nonnegative constraint is applied when each stage-derived `c`
   candidate is produced, before the two candidates are averaged.
