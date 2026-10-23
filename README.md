```bash
python train_tailfm.py --data data/returns_clean.csv --outdir runs/cond_all --train-all --cond-valor 1234567 --gen-cond 0.8

python run_backtest.py --data data/returns_clean.csv --gen runs/cond_all/generated_windows_u0.80.npy --horizon 1 --cond-band 0.7 0.9

python evaluate.py --data data/returns_clean.csv --gen runs/cond_all/generated_windows_u0.80.npy --no-marginals

python cond_check.py --data data/returns_clean.csv --run runs/cond_all



python train_tailfm.py --data data/returns_clean.csv --outdir runs/cond_multi --train-all --cond-valor 4155001 --gen-cond 0.3 0.6 0.9

python evaluate.py --data data/returns_clean.csv --gen runs/cond_multi/generated_windows_u0.30.npy --no-marginals
python evaluate.py --data data/returns_clean.csv --gen runs/cond_multi/generated_windows_u0.60.npy --no-marginals
python evaluate.py --data data/returns_clean.csv --gen runs/cond_multi/generated_windows_u0.90.npy --no-marginals

python run_backtest.py --data data/returns_clean.csv --gen runs/cond_multi/generated_windows_u0.30.npy --horizon 1 --cond-band 0.25 0.35 --log runs/cond_multi/backtest_u0.30.log
python run_backtest.py --data data/returns_clean.csv --gen runs/cond_multi/generated_windows_u0.60.npy --horizon 1 --cond-band 0.55 0.65 --log runs/cond_multi/backtest_u0.60.log
python run_backtest.py --data data/returns_clean.csv --gen runs/cond_multi/generated_windows_u0.90.npy --horizon 1 --cond-band 0.85 0.95 --log runs/cond_multi/backtest_u0.90.log

python cond_check.py --data data/returns_clean.csv --run runs/cond_multi --levels 0.3 0.6 0.9 --num 20000 --save
python evaluate.py --data data/returns_clean.csv --gen runs/cond_multi/generated_windows_u0.90.npy --no-marginals


python generate.py --data data/returns_clean.csv --run runs/u090 --levels 0.9 --no-save
python generate.py --data data/returns_clean.csv --run runs/u090 --levels 0.9 --no-save --no-recalibrate


python evaluate.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy --cond-level 0.9 --no-figures --no-novelty --no-report --weights $(python -c "print(','.join(['-0.00404858'] * 247))") --log runs/u090/evaluate_h1_short.log


```
