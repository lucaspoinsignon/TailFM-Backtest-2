```bash


cd /repos/quail
python 02_returns.py --data data/prices.csv --mm-data data/prices_panel_2000.csv --out data/returns_mm.csv

python train_tailfm.py --data data/returns_mm.csv --cond-valor 4155001 --train-all --steps 40000 --outdir runs/mm090
python generate.py --data data/returns_mm.csv --run runs/mm090 --levels 0.9
python mm_check.py --data data/returns_mm.csv --gen runs/mm090/generated_windows_u0.90.npy

python train_tailfm.py --data data/returns_mm.csv --train-all --steps 40000 --outdir runs/mm_uncond
python generate.py --data data/returns_mm.csv --run runs/mm_uncond
python mm_check.py --data data/returns_mm.csv --gen runs/mm_uncond/generated_windows.npy


```
