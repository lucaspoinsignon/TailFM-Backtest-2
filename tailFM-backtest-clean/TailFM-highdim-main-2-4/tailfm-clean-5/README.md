# TailFM conditional — commands

```bash
A="--data data/returns_clean.csv --cond-valor 4155001 --steps 40000 --d-model 768"
D="--data data/returns_clean.csv"
```

## Train

```bash
# unconditional (original): no --gen-cond
python train_tailfm.py $D --steps 40000 --d-model 768 --outdir runs/uncond

# conditional: --gen-cond with the u value(s) you choose
python train_tailfm.py $A --train-all --gen-cond 0.75 --outdir runs/u075

# all days
python train_tailfm.py $A --train-all --gen-cond 0.3 0.6 0.9 --outdir runs/all

# only a period
python train_tailfm.py $A --train-period 2005-01-01 2015-12-31 --gen-cond 0.9 --outdir runs/p0515

# everything except a period
python train_tailfm.py $A --exclude-period 2020-01-01 2020-12-31 --gen-cond 0.9 --outdir runs/ex2020

# a period minus a sub-period
python train_tailfm.py $A --train-period 2005-01-01 2019-12-31 --exclude-period 2008-01-01 2008-12-31 --gen-cond 0.9 --outdir runs/p0519_ex08

# band generation instead of point (default point)
python train_tailfm.py $A --train-all --gen-cond 0.9 --gen-mode band --outdir runs/all_band
python train_tailfm.py $A --train-all --gen-cond 0.9 --gen-mode band --band-width 0.2 --outdir runs/all_band20
```

## Regenerate from a saved model (no retraining)

```bash
# other levels, point / band
python cond_check.py $D --run runs/all --levels 0.5 0.8 --num 20000 --save
python cond_check.py $D --run runs/all --levels 0.5 0.8 --mode band --num 20000 --save

# matched to test days: a band (in-sample)
python cond_check.py $D --run runs/all --match --cond-band 0.85 0.95 --num 20000 --save

# matched to test days: a left-out period (out-of-sample)
python cond_check.py $D --run runs/ex2020 --match --test-period 2020-01-01 2020-12-31 --num 20000 --save

# matched to test days: a left-out period and a band
python cond_check.py $D --run runs/ex2020 --match --test-period 2020-01-01 2020-12-31 --cond-band 0.85 1.0 --num 20000 --save
```

## Backtest (test days and band default to the sample's .json)

```bash
python run_backtest.py $D --gen runs/all/generated_windows_u0.90.npy --horizon 1 --log runs/all/bt_u0.90.log
python run_backtest.py $D --gen runs/all/generated_windows_match_u0.85-0.95.npy --horizon 1 --log runs/all/bt_band.log
python run_backtest.py $D --gen runs/ex2020/generated_windows_match_2020-01-01_2020-12-31.npy --horizon 1 --log runs/ex2020/bt_2020.log

# explicit test period / band (override the .json; --cond-band 0 1 removes the band)
python run_backtest.py $D --gen runs/ex2020/generated_windows_u0.90.npy --test-period 2020-01-01 2020-12-31 --cond-band 0.85 0.95 --horizon 1 --log runs/ex2020/bt_u0.90_2020.log

# unconditional
python run_backtest.py $D --gen runs/uncond/generated_windows.npy --horizon 1
```

## Evaluate (--no-marginals for runs without held-out blocks)

```bash
python evaluate.py $D --gen runs/all/generated_windows_u0.90.npy --no-marginals
python evaluate.py $D --gen runs/ex2020/generated_windows_match_2020-01-01_2020-12-31.npy --no-marginals
python evaluate.py $D --gen runs/ex2020/generated_windows_u0.90.npy --test-period 2020-01-01 2020-12-31 --no-marginals
python evaluate.py $D --gen runs/uncond/generated_windows.npy
```

## Scorecard (several samples, same real days)

```bash
python compare_band.py $D --run runs/all --cond-band 0.85 0.95 \
  --gen point=runs/all/generated_windows_u0.90.npy \
  --gen band=runs/all_band/generated_windows_u0.85-0.95.npy \
  --gen match=runs/all/generated_windows_match_u0.85-0.95.npy

python compare_band.py $D --run runs/ex2020 --test-period 2020-01-01 2020-12-31 --cond-band 0.85 1.0 \
  --gen match=runs/ex2020/generated_windows_match_u0.85-1.00_2020-01-01_2020-12-31.npy
```
