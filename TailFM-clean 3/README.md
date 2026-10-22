# TailFM: regime-conditional scenario generator

```
CSV  ->  plots  ->  train_tailfm.py  ->  generate.py  ->  evaluate.py / run_backtest.py
                                                      ->  make_folds.py -> folds.sh -> pooled_ci.py
```

The model is `tailfm/` (EVT marginals + Student-t flow matching) with `evt_shrink.py`.
Training only fits and saves it; every sample is drawn afterwards by `generate.py`, so one
trained run serves any regime level.

Regime u = training percentile of the EWMA vol of the `--cond-valor` series. A **value**
U always means the days with u in [U - 0.03, U + 0.03]; a **band** is given as LO HI.

## 0. Data and plots

```bash
python 01_panel.py --raw data/raw.csv --out data/prices.csv --currency USD
python 02_returns.py --data data/prices.csv --out data/returns.csv
python drop_valors.py --data data/returns.csv --out data/returns_clean.csv --valors 4155686,4155690
python check.py --data data/returns_clean.csv
python 03_plot.py --data data/returns_clean.csv --out fig/returns.png
python 04_analyse.py --data data/returns_clean.csv --out fig/dependence.png
```

`check.py` exits non-zero if training would break on the CSV.

## 1. Train

```bash
# every day
python train_tailfm.py --data data/returns_clean.csv --cond-valor 4155001 --steps 40000 --d-model 768 --train-all --outdir runs/u090

# keep 20% of the u ~ 0.9 days for testing (whole episodes spread over time, +-25 days around each)
python train_tailfm.py --data data/returns_clean.csv --cond-valor 4155001 --steps 40000 --d-model 768 --holdout-level 0.9 --outdir runs/ho090
python train_tailfm.py --data data/returns_clean.csv --cond-valor 4155001 --steps 40000 --d-model 768 --holdout-band 0.85 1.0 --holdout-frac 0.3 --outdir runs/ho085

# leave a period out
python train_tailfm.py --data data/returns_clean.csv --cond-valor 4155001 --steps 40000 --d-model 768 --exclude-period 2020-01-01 2020-12-31 --outdir runs/ex2020

# only a period, minus a sub-period
python train_tailfm.py --data data/returns_clean.csv --cond-valor 4155001 --steps 40000 --d-model 768 --train-period 2005-01-01 2019-12-31 --exclude-period 2008-01-01 2008-12-31 --outdir runs/p0519_ex08

# unconditional (no --cond-valor)
python train_tailfm.py --data data/returns_clean.csv --steps 40000 --d-model 768 --outdir runs/uncond
```

Without `--train-all` / `--train-period` / `--exclude-period` / `--holdout-*`, random
purged blocks are held out (`splits.py`). The run folder gets `split.json`,
`model_ema.pt`, `model_config.json`, `marginals.pkl`, `cond_u.npy`, `cond_sigma_ref.npy`,
`report.log`, and `holdout.json` (the held-out periods) for a `--holdout-*` run.

## 2. Generate

```bash
# every window at u = 0.8, and at u = 0.9
python generate.py --data data/returns_clean.csv --run runs/u090 --levels 0.8 0.9

# u uniform in a band
python generate.py --data data/returns_clean.csv --run runs/u090 --band 0.87 0.93

# u of real days: a value or band, optionally restricted to the held-out days or a period
python generate.py --data data/returns_clean.csv --run runs/u090 --match --cond-level 0.9
python generate.py --data data/returns_clean.csv --run runs/ho090 --match --holdout --cond-level 0.9
python generate.py --data data/returns_clean.csv --run runs/ex2020 --match --test-period 2020-01-01 2020-12-31 --cond-band 0.85 1.0

# unconditional run
python generate.py --data data/returns_clean.csv --run runs/uncond

# table only (VaR across levels), nothing saved
python generate.py --data data/returns_clean.csv --run runs/u090 --levels 0.7 0.75 0.8 0.85 0.9 0.95 --no-save
```

Files: `generated_windows_u0.90.npy`, `generated_windows_u0.87-0.93.npy`,
`generated_windows_match_<selection>.npy`, `generated_windows.npy`. Each conditional
sample has a `.json` sidecar with the real days that validate it (a point sample at U:
u in [U - 0.03, U + 0.03]); evaluate and backtest use it by default. Defaults (`--num
20000`, the run's seed and ODE steps) reproduce the old `train_tailfm.py` samples bit for
bit. The printed table compares each sample's equal-weight 1-day P&L (mean, sd, VaR95,
VaR99, ES97.5) with the real days of its band.

## 3. Evaluate

```bash
# vs the training windows of the sample's band (from its sidecar)
python evaluate.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy

# vs every training window (entire dataset)
python evaluate.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy --no-band

# vs a value or a band
python evaluate.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy --cond-level 0.9
python evaluate.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy --cond-band 0.85 0.95

# vs the held-out days of the run
python evaluate.py --data data/returns_clean.csv --gen runs/ho090/generated_windows_u0.90.npy --holdout --cond-level 0.9

# vs a period (trained on or not), or without a period
python evaluate.py --data data/returns_clean.csv --gen runs/ex2020/generated_windows_u0.90.npy --test-period 2020-01-01 2020-12-31
python evaluate.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy --exclude-period 2008-01-01 2009-12-31

# unconditional sample: all training windows
python evaluate.py --data data/returns_clean.csv --gen runs/uncond/generated_windows.npy
```

`--horizon 1` sets the risk table to 1 day (default: the split's horizon). The marginal
PIT section is skipped when the run has no held-out rows.

## 4. Backtest

```bash
# days of the sample's band (from its sidecar)
python run_backtest.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy --horizon 1

# a value, a band, or every day
python run_backtest.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy --horizon 1 --cond-level 0.9
python run_backtest.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy --horizon 1 --cond-band 0.85 0.95
python run_backtest.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy --horizon 1 --no-band

# out of sample: the run's held-out days
python run_backtest.py --data data/returns_clean.csv --gen runs/ho090/generated_windows_u0.90.npy --horizon 1 --holdout --cond-level 0.9

# a given period, or all data without a period
python run_backtest.py --data data/returns_clean.csv --gen runs/ex2020/generated_windows_u0.90.npy --horizon 1 --test-period 2020-01-01 2020-12-31 --cond-band 0.85 1.0
python run_backtest.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy --horizon 1 --exclude-period 2020-01-01 2020-12-31

# reproduce the papers' tables
python run_backtest.py --self-test
```

The flags combine (period, exclusion, hold-out, value or band). Default test days: the
sample's sidecar, else the split's held-out blocks, else every day. The log states
whether the test days were training days (in-sample) or not (out-of-sample). Kupiec PF /
TUFF at `--var-levels` (0.95, 0.99) and Acerbi-Szekely Z1/Z2/Z3 at `--es-levels`
(0.975, 0.99).

## 5. K-fold out-of-sample backtest

```bash
python make_folds.py --run runs/u090 --band 0.87 0.93 --k 5
nohup bash folds.sh > folds.out 2>&1 &
python pooled_ci.py
python pooled_ci.py --same runs/u090/generated_windows_u0.90.npy
python pooled_ci.py --level 0.80
```

`make_folds.py` groups the band episodes into 5 folds and writes `folds.sh`: per fold,
train without its episodes (+-25 days), generate u = 0.8 and 0.9, backtest on its
episodes. Training flags are copied from the run's `report.log`. `pooled_ci.py` pools all
folds into one out-of-sample test and scans a scale c of the predictive law (accepted c
containing 1 = calibrated); `--same` scores the same days with the in-sample model,
`--level 0.80` checks that the wrong regime is rejected.

## Regime analysis

```bash
python regime_periods.py --data data/returns_clean.csv --run runs/u090 --levels 0.8 0.9
python diag_band.py --data data/returns_clean.csv --run runs/u090 --valor 4155001 --band 0.85 0.95
python compare_band.py --data data/returns_clean.csv --run runs/u090 --cond-band 0.87 0.93 --gen point=runs/u090/generated_windows_u0.90.npy --gen match=runs/u090/generated_windows_match_u0.87-0.93.npy
python vol_corr_check.py --data data/returns_clean.csv --outdir fig/vol_corr
python 06_compare.py --data data/returns_clean.csv --run runs/u090 --gen runs/u090/generated_windows_u0.90.npy --out fig/windows.png
```

- `regime_periods.py`: which days are in a band, as episodes and per year.
- `diag_band.py`: onsets vs aftermaths in a band.
- `compare_band.py`: scorecard of several samples against the same real days.
- `vol_corr_check.py`: EWMA vol regimes and correlations.
- `06_compare.py`: generated vs real windows, side by side.

## Other

`baselines/` and `run_baselines.py` (TimeVAE, TimeGAN, Tail-GAN, scored with the same
code) are kept but are not part of the pipeline above.
