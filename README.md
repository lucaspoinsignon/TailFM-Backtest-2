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


grep "# command" runs/ho090/report.log

nohup python momentum/train_mom.py --data data/returns_clean.csv --cond-valor 4155001 --steps 40000 --d-model 768 --holdout-level 0.9 --outdir runs/ho090_mom > runs/ho090_mom.out 2>&1 &
tail -f runs/ho090_mom.out

python momentum/generate_mom.py --data data/returns_clean.csv --run runs/ho090_mom --holdout --cond-level 0.9 --direction all rising falling
python momentum/generate_mom.py --data data/returns_clean.csv --run runs/ho090 --holdout --cond-level 0.9 --direction all rising falling

python momentum/backtest_mom.py --data data/returns_clean.csv --run level=runs/ho090 --run momentum=runs/ho090_mom --holdout --cond-level 0.9 --direction all rising falling

python momentum/generate_mom.py --data data/returns_clean.csv --run runs/ho090_mom --cond-level 0.9 --direction rising falling --no-save
python momentum/generate_mom.py --data data/returns_clean.csv --run runs/ho090 --cond-level 0.9 --direction rising falling --no-save

python evaluate.py --data data/returns_clean.csv --gen runs/u090/generated_windows_u0.90.npy --cond-level 0.9 --no-figures --no-novelty


```
