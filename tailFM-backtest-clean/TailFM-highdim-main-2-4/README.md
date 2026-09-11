# TailFM

The code and the commands to run it are in [`tailfm-clean-5/`](tailfm-clean-5/README.md).

Subsample a panel down to 30 random columns:

```bash
python -c "
import numpy as np, pandas as pd
df = pd.read_csv('data/returns_clean.csv', index_col=0, parse_dates=True)
sel = np.random.default_rng(0).choice(df.columns, 30, replace=False)
out = df.loc[:, df.columns.isin(sel)]
out.to_csv('data/returns_30.csv')
print(out.shape, list(out.columns))
"
```
