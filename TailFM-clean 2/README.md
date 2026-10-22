# TailFM: regime-conditional scenario generator

```
CSV  ->  plots  ->  train_tailfm.py  ->  generate.py  ->  evaluate.py / run_backtest.py
                                                      ->  make_folds.py -> folds.sh -> pooled_ci.py
```

The model is `tailfm/` (EVT marginals + Student-t flow matching) with `evt_shrink.py`.
Training only fits and saves it; every sample is drawn afterwards by `generate.py`, so one
trained run serves any regime level.

```bash
D="--data data/returns_clean.csv"
A="$D --cond-valor 4155001 --steps 40000 --d-model 768"
```

## 0. Data and plots

```bash
python 01_panel.py   --raw data/raw.csv --out data/prices.csv --currency USD
python 02_returns.py --data data/prices.csv --out data/returns.csv
python drop_valors.py --data data/returns.csv --out data/returns_clean.csv --valors 4155686,4155690
python check.py      --data data/returns_clean.csv            # exits non-zero if training would break
python 03_plot.py    --data data/returns_clean.csv --out fig/returns.png
python 04_analyse.py --data data/returns_clean.csv --out fig/dependence.png
```

## 1. Train

```bash
python train_tailfm.py $A --train-all --outdir runs/u090                       # every day
python train_tailfm.py $A --exclude-period 2020-01-01 2020-12-31 --outdir runs/ex2020
python train_tailfm.py $A --train-period 2005-01-01 2019-12-31 \
    --exclude-period 2008-01-01 2008-12-31 --outdir runs/p0519_ex08
python train_tailfm.py $D --steps 40000 --d-model 768 --outdir runs/uncond      # unconditional
```

`--cond-valor` turns regime conditioning on (u = training percentile of that VALOR's EWMA
vol). Without `--train-all` / `--train-period` / `--exclude-period`, random purged blocks
are held out (`splits.py`). The run folder gets `split.json`, `model_ema.pt`,
`model_config.json`, `marginals.pkl`, `cond_u.npy`, `cond_sigma_ref.npy`, `report.log`.

## 2. Generate

```bash
python generate.py $D --run runs/u090 --levels 0.8 0.9                    # point: every window at u
python generate.py $D --run runs/u090 --levels 0.9 --mode band --band-width 0.06   # u uniform in the band
python generate.py $D --run runs/u090 --match --cond-band 0.87 0.93       # u of those real days
python generate.py $D --run runs/ex2020 --match --test-period 2020-01-01 2020-12-31 --cond-band 0.85 1.0
python generate.py $D --run runs/uncond                                   # unconditional run
python generate.py $D --run runs/u090 --levels 0.70 0.75 0.80 0.85 0.90 0.95 --no-save   # VaR(u) table only
```

Files: `generated_windows_u0.90.npy`, `generated_windows_u0.85-0.95.npy`,
`generated_windows_match_<selection>.npy`, `generated_windows.npy`, each conditional one
with a `.json` sidecar (the band / days that validate it). Defaults: `--num 20000`, the
run's seed and ODE steps, which reproduce what the old `train_tailfm.py` generated, bit
for bit. A table compares each sample's equal-weight 1-day P&L (mean, sd, VaR95, VaR99,
ES97.5) with the real days of its band.

## 3. Evaluate

```bash
G=runs/u090/generated_windows_u0.90.npy
python evaluate.py $D --gen $G                          # vs training windows with u in the sample's band
python evaluate.py $D --gen $G --no-band                # vs every training window (whole dataset)
python evaluate.py $D --gen $G --cond-band 0.87 0.93    # vs an explicit band (or --cond-level 0.9 --band-width 0.06)
python evaluate.py $D --gen $G --test-period 2020-01-01 2020-12-31   # windows in a period, trained on or not
python evaluate.py $D --gen $G --exclude-period 2008-01-01 2009-12-31
python evaluate.py $D --gen runs/uncond/generated_windows.npy        # unconditional: all training windows
```

`--horizon 1` sets the risk table to 1 day (default: the split's horizon). The marginal
PIT section is skipped automatically when the run has no held-out rows.

## 4. Backtest

```bash
G=runs/u090/generated_windows_u0.90.npy
python run_backtest.py $D --gen $G --horizon 1                                  # days of the sample's band
python run_backtest.py $D --gen $G --horizon 1 --cond-band 0.87 0.93            # a band
python run_backtest.py $D --gen $G --horizon 1 --cond-level 0.9 --band-width 0.06   # a value +- w/2
python run_backtest.py $D --gen $G --horizon 1 --no-band                        # every day, any u
python run_backtest.py $D --gen runs/ex2020/generated_windows_u0.90.npy --horizon 1 \
    --test-period 2020-01-01 2020-12-31 --cond-band 0.85 1.0                    # a given period
python run_backtest.py $D --gen $G --horizon 1 --exclude-period 2020-01-01 2020-12-31   # all data without it
python run_backtest.py $D --gen $G --horizon 1 --cond-band 0.87 0.93 --demean   # zero-mean scenarios
python run_backtest.py --self-test                                              # reproduce the papers' tables
```

Flags combine (period + exclusion + band). Default test days: the sample's sidecar, else
the split's held-out blocks, else every day. The log states whether the test days were
training days (in-sample) or not (out-of-sample). Kupiec PF / TUFF at `--var-levels`
(0.95, 0.99) and Acerbi-Szekely Z1/Z2/Z3 at `--es-levels` (0.975, 0.99).

## 5. K-fold out-of-sample backtest

```bash
python make_folds.py --run runs/u090 --band 0.87 0.93 --k 5    # plan: folds.sh + folds.json
nohup bash folds.sh > folds.out 2>&1 &                          # per fold: train, generate, backtest
python pooled_ci.py                                             # pooled c-scan over all held-out band days
python pooled_ci.py --same runs/u090/generated_windows_u0.90.npy    # same periods, in-sample model
python pooled_ci.py --level 0.80                                    # wrong-regime check
```

Each fold leaves out a group of band episodes (plus 25 days on each side), trains
without them and is tested on them; the training flags are copied from the run's
`report.log`. `pooled_ci.py` uses zero-mean scenarios and scans the scale c of the
predictive law: the accepted c contain 1 if the model is calibrated.

## Regime analysis

```bash
python regime_periods.py $D --run runs/u090 --levels 0.8 0.9 --band-width 0.06   # band days: episodes, per year
python diag_band.py      $D --run runs/u090 --valor 4155001 --band 0.85 0.95      # onsets vs aftermaths
python compare_band.py   $D --run runs/u090 --cond-band 0.87 0.93 \
    --gen point=runs/u090/generated_windows_u0.90.npy \
    --gen match=runs/u090/generated_windows_match_u0.87-0.93.npy                  # scorecard vs the same days
python vol_corr_check.py $D --outdir fig/vol_corr                                 # EWMA vol and correlation regimes
python 06_compare.py     $D --run runs/u090 --gen runs/u090/generated_windows_u0.90.npy --out fig/windows.png
```

## Other

`baselines/` and `run_baselines.py` (TimeVAE, TimeGAN, Tail-GAN, scored with the same
code) are kept but are not part of the pipeline above.
