```bash


python - <<'EOF'
import numpy as np, pandas as pd
for f in ["data/returnscomp.csv", "data/returns05.csv"]:
    d = pd.to_datetime(pd.read_csv(f, usecols=[0]).iloc[:, 0]).values.astype("datetime64[D]")
    k = np.busday_count(d[:-1], d[1:])
    print(f, "| returns spanning > 10 weekdays:", ", ".join(f"{d[i]} -> {d[i + 1]} ({k[i]})" for i in np.flatnonzero(k > 10)))
EOF
python panel_span.py --prices data/prices.csv --returns data/returns05.csv   # plus the flags you gave 02_returns.py

```
