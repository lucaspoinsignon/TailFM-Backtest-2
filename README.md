```bash

python check.py --data data/returns06.csv       

python train_tailfm.py --data data/returns06.csv --train-all --outdir runs/r06
python generate.py --data data/returns06.csv --run runs/r06
python evaluate.py --data data/returns06.csv --gen runs/r06/generated_windows.npy
python run_backtest.py --data data/returns06.csv --gen runs/r06/generated_windows.npy --horizon 1,10
python stylized_facts.py --data data/returns06.csv --gen runs/r06/generated_windows.npy

python train_tailfm.py --data data/returns06.csv --cond-valor 4155001 --train-all --outdir runs/r06u
python generate.py --data data/returns06.csv --run runs/r06u --band 0.87 0.93
python evaluate.py --data data/returns06.csv --gen runs/r06u/generated_windows_u0.87-0.93.npy
python run_backtest.py --data data/returns06.csv --gen runs/r06u/generated_windows_u0.87-0.93.npy --horizon 1,10
python stylized_facts.py --data data/returns06.csv --gen runs/r06u/generated_windows_u0.87-0.93.npy
python regime_calibration.py --data data/returns06.csv --run runs/r06u

```
