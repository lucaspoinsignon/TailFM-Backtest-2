```bash
python train_tailfm.py --data data/returns_clean.csv --outdir runs/cond_all --train-all --cond-valor 1234567 --gen-cond 0.8

python run_backtest.py --data data/returns_clean.csv --gen runs/cond_all/generated_windows_u0.80.npy --horizon 1 --cond-band 0.7 0.9

python evaluate.py --data data/returns_clean.csv --gen runs/cond_all/generated_windows_u0.80.npy --no-marginals

python cond_check.py --data data/returns_clean.csv --run runs/cond_all

```
