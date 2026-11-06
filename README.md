```bash


python - <<'EOF'
import numpy as np, pandas as pd
d = pd.to_datetime(pd.read_csv("data/returns06.csv", usecols=[0]).iloc[:, 0]).values.astype("datetime64[D]")
k = np.busday_count(d[:-1], d[1:])
print(d[0], "->", d[-1], len(d), "|", " ".join(f"{a}:{b}" for a, b in zip(*np.unique(k, return_counts=True))))
print("spans > 10 weekdays:", ", ".join(f"{d[i]} -> {d[i + 1]} ({k[i]})" for i in np.flatnonzero(k > 10)))
EOF

```
