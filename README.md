```bash


python - <<'EOF'
import numpy as np, pandas as pd
for f in ["data/returnscomp.csv", "data/returns05.csv"]:
    d = pd.to_datetime(pd.read_csv(f, usecols=[0]).iloc[:, 0]).values.astype("datetime64[D]")
    k = np.busday_count(d[:-1], d[1:])
    print(f, d[0], "->", d[-1], len(d), "returns | weekdays spanned:",
          " ".join(f"{a}:{b}" for a, b in zip(*np.unique(k, return_counts=True))))
EOF

```
