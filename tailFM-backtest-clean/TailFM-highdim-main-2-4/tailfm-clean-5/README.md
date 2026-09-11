# TailFM

Tail-aware flow matching for multivariate return generation, targeted at VaR / CVaR.

```bash
pip install numpy pandas scipy matplotlib torch
```

## 1. Data

```bash
python 01_panel.py    --raw data/raw.csv --out data/prices.csv --currency USD
python 02_returns.py  --data data/prices.csv --out data/returns.csv
python check.py       --data data/returns.csv
python drop_valors.py --data data/returns.csv --out data/returns_clean.csv \
    --valors 4155686,4155690,4157124
```

## 2. Plots

```bash
python 03_plot.py    --data data/returns.csv --out fig/returns.png
python 04_analyse.py --data data/returns.csv --out fig/dependence.png
```

## 3. Train and generate

```bash
python train_tailfm.py --data data/returns_clean.csv --outdir runs/final
```

Defaults: `--nu 5 --q-tail 0.05 --pos-std 0.1 --n 24 --test-frac 0.2 --horizon 10
--steps 20000 --gen 20000 --d-model 512 --ode-steps 100`. Writes `split.json`,
`generated_windows.npy`, `model_ema.pt`, `marginals.pkl` and `report.log` into
`--outdir`.

## 4. Evaluate

```bash
python evaluate.py --data data/returns_clean.csv --gen runs/final/generated_windows.npy
```

Several generators at once:

```bash
python evaluate.py --data data/returns_clean.csv --split runs/final/split.json \
    --gen tailfm=runs/final/generated_windows.npy \
    --gen timevae=runs/baselines/gen_timevae.npy \
    --outdir runs/compare --pair-select spread
```

## 5. Backtest

```bash
python run_backtest.py --data data/returns_clean.csv --gen runs/final/generated_windows.npy
python run_backtest.py --self-test
```

## 6. Baselines

```bash
python run_baselines.py --data data/returns_clean.csv \
    --split runs/final/split.json \
    --tailfm-gen runs/final/generated_windows.npy --outdir runs/baselines
```

## 7. Compare runs

```bash
python summarize_runs.py --data data/returns_clean.csv --split runs/final/split.json \
    runs/final runs/other
python 06_compare.py --data data/returns_clean.csv --split runs/final/split.json \
    --run runs/final --dim 3
```
