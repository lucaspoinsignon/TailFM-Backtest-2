```bash
python factor_whatif.py --data data/returns_clean.csv --gen runs/all090/generated_windows_u0.90.npy
mkdir -p runs/all090/sf_fx
python stylized_facts.py --data data/returns_clean.csv --gen runs/all090/generated_windows_u0.90_fx.npy --outdir runs/all090/sf_fx --no-assets-summary
python run_backtest.py --data data/returns_clean.csv --gen base=runs/all090/generated_windows_u0.90.npy --gen fx=runs/all090/generated_windows_u0.90_fx.npy --horizon 1 --log runs/all090/bt_fx.log

# 1b conditional, out-of-sample (each fold's target uses its own training days only)
for k in 1 2 3 4 5; do python factor_whatif.py --data data/returns_clean.csv --gen runs/fold$k/generated_windows_u0.90.npy; done
python pooled_ci.py | tee pooled_base.txt
python pooled_ci.py --suffix _fx | tee pooled_fx.txt

# 1c unconditional (all days, in-sample)
python factor_whatif.py --data data/returns_clean.csv --gen runs/uncond/generated_windows.npy
mkdir -p runs/uncond/sf_fx
python stylized_facts.py --data data/returns_clean.csv --gen runs/uncond/generated_windows_fx.npy --outdir runs/uncond/sf_fx --no-assets-summary
python run_backtest.py --data data/returns_clean.csv --gen base=runs/uncond/generated_windows.npy --gen fx=runs/uncond/generated_windows_fx.npy --horizon 1 --log runs/uncond/bt_fx.log



grep "# command" runs/uncond/report.log runs/all090/report.log




# 2a unconditional
python train_tailfm.py --data data/returns_clean.csv --train-all --steps 40000 --source skew --outdir runs/uncond_skew
python generate.py --data data/returns_clean.csv --run runs/uncond_skew
python stylized_facts.py --data data/returns_clean.csv --gen runs/uncond_skew/generated_windows.npy
python run_backtest.py --data data/returns_clean.csv --gen base=runs/uncond/generated_windows.npy --gen skew=runs/uncond_skew/generated_windows.npy --horizon 1 --log runs/uncond_skew/bt_compare.log

# 2b conditional, in-sample
python train_tailfm.py --data data/returns_clean.csv --cond-valor 4155001 --train-all --steps 40000 --source skew --outdir runs/all090_skew
python generate.py --data data/returns_clean.csv --run runs/all090_skew --levels 0.9
python stylized_facts.py --data data/returns_clean.csv --gen runs/all090_skew/generated_windows_u0.90.npy
python run_backtest.py --data data/returns_clean.csv --gen base=runs/all090/generated_windows_u0.90.npy --gen skew=runs/all090_skew/generated_windows_u0.90.npy --horizon 1 --log runs/all090_skew/bt_compare.log

# 2c conditional, out-of-sample (same folds as the base, new folders runs/skewfold1..5)
python make_folds.py --run runs/all090_skew --prefix runs/skewfold --name folds_skew
bash folds_skew.sh
python pooled_ci.py --folds folds_skew.json | tee pooled_skew.txt


```
