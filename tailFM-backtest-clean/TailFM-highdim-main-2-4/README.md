# TailFM

The code and the commands to run it are in [`tailfm-clean-5/`](tailfm-clean-5/README.md).

Subsample a panel down to 30 random columns:

```bash
python make_hs.py --data data/returns_clean.csv --run runs/final3 --out runs/hs
python run_backtest.py --data data/returns_clean.csv \
    --gen tailfm=runs/final3/generated_windows.npy \
    --gen hs=runs/hs/generated_windows.npy
```
