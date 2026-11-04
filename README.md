```bash


python check.py --data data/returnscomp.csv


python train_tailfm.py --data data/returnscomp.csv --train-all --outdir runs/all


python generate.py --data data/returnscomp.csv --run runs/all



python screen_valors.py --data data/returnscomp.csv --out fig/screen
python class_check.py --data data/returnscomp.csv --gen runs/all/generated_windows.npy \
    --screen fig/screen/screen.csv --pairs fig/screen/duplicates.txt
python position_check.py --data data/returnscomp.csv --gen runs/all/generated_windows.npy
python run_backtest.py --data data/returnscomp.csv --gen runs/all/generated_windows.npy --horizon 1
python evaluate.py --data data/returnscomp.csv --gen runs/all/generated_windows.npy


```
