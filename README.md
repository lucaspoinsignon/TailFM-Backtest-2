```bash


python screen_valors.py --data data/returnscomp.csv --out fig/screen --plot 4157818,4155464,4157572,4157300,102695251,4155895
python position_check.py --data data/returns_clean.csv --gen runs/clean_data/generated_windows.npy \
    --pairs 4155867:4157572,4155894:4157300,4042177:4157572,4157300:4157876
python drop_valors.py --data data/returnscomp.csv --out data/returns_v2.csv --file fig/screen/drop.txt
```
