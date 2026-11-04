```bash


python train_tailfm.py --data data/returnscomp.csv --cond-valor 4155001 --train-all --outdir runs/all_u
python generate.py --data data/returnscomp.csv --run runs/all_u --band 0.87 0.93
python stylized_facts.py --data data/returnscomp.csv --gen runs/all_u/generated_windows_u0.87-0.93.npy
python run_backtest.py --data data/returnscomp.csv --gen runs/all_u/generated_windows_u0.87-0.93.npy --horizon 1,10
python stylized_facts.py --data data/returnscomp.csv --gen runs/all/generated_windows.npy
python run_backtest.py --data data/returnscomp.csv --gen runs/all/generated_windows.npy --horizon 10 --log runs/all/backtest_h10.log


```
