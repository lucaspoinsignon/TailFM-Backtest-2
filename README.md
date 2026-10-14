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


```

```bash
python diag_band.py --data data/returns_clean.csv --run runs/cond_multi2 --valor 4155001 --band 0.85 0.95

A="--data data/returns_clean.csv --train-all --gen-cond 0.9"
python train_tailfm.py $A --cond-valor 4155001                     --outdir runs/exp_base
python train_tailfm.py $A --cond-valor 4155001 --seed 1            --outdir runs/exp_base_s1
python train_tailfm.py $A --cond-valor 4155001 --cond-momentum 10  --outdir runs/exp_mom
python train_tailfm.py $A --cond-valor 4155001 --oversample 0.85 5 --outdir runs/exp_over
python train_tailfm.py $A --cond-valor EW                          --outdir runs/exp_ew

for m in base base_s1 mom over ew; do
  python cond_check.py --data data/returns_clean.csv --run runs/exp_$m \
    --ref-u runs/exp_base/cond_u.npy --band 0.85 0.95 --num 20000 --save
done

M=generated_windows_match_0.85-0.95.npy
python compare_band.py --data data/returns_clean.csv --ref-u runs/exp_base/cond_u.npy --band 0.85 0.95 \
  --gen point=runs/exp_base/generated_windows_u0.90.npy --gen band=runs/exp_base/$M \
  --gen seed1=runs/exp_base_s1/$M --gen momentum=runs/exp_mom/$M \
  --gen oversample=runs/exp_over/$M --gen ew_vol=runs/exp_ew/$M
```
