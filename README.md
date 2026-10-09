```bash
python train_tailfm_eps.py --data data/returns_clean.csv --outdir runs/eps_window --mix-dim window --steps 40000 --d-model 768
python train_tailfm_eps.py --data data/returns_clean.csv --outdir runs/eps_none   --mix-dim none   --steps 40000 --d-model 768

python eval_eps.py --data data/returns_clean.csv --gen eps_window=runs/eps_window --gen eps_none=runs/eps_none --orig runs/final3

```
