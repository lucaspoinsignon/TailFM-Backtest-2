```bash


python screen_valors.py --data data/returnscomp.csv --reference data/returns_clean.csv --out fig/screen

W="$(cat fig/screen/weights_reference.txt)"
python run_backtest.py --data data/returnscomp.csv --gen runs/complete/generated_windows.npy --horizon 1 --weights "$W" --log runs/complete/backtest_ref247.log
python evaluate.py --data data/returnscomp.csv --gen runs/complete/generated_windows.npy --weights "$W" --no-report --no-marginals --no-novelty --no-figures --log runs/complete/evaluate_ref247.log

```
