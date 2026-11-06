```bash

python check.py --data data/returns06.csv       

python train_tailfm.py --data data/returns06_clean.csv --train-all --outdir runs/r06_clean
python generate.py --data data/returns06_clean.csv --run runs/r06_clean
python evaluate.py --data data/returns06_clean.csv --gen runs/r06_clean/generated_windows.npy
python run_backtest.py --data data/returns06_clean.csv --gen runs/r06_clean/generated_windows.npy --horizon 1,10
python stylized_facts.py --data data/returns06_clean.csv --gen runs/r06_clean/generated_windows.npy

python train_tailfm.py --data data/returns06_clean.csv --cond-valor 4155001 --train-all --outdir runs/r06u_clean
python generate.py --data data/returns06_clean.csv --run runs/r06u_clean --band 0.87 0.93
python evaluate.py --data data/returns06_clean.csv --gen runs/r06u_clean/generated_windows_u0.87-0.93.npy
python run_backtest.py --data data/returns06_clean.csv --gen runs/r06u_clean/generated_windows_u0.87-0.93.npy --horizon 1,10
python stylized_facts.py --data data/returns06_clean.csv --gen runs/r06u_clean/generated_windows_u0.87-0.93.npy
python regime_calibration.py --data data/returns06_clean.csv --run runs/r06u_clean


```
