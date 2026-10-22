"""Out-of-sample coverage tests on the purged held-out blocks of `split.json`.

  kupiec       Kupiec (1995) TUFF and PF likelihood-ratio tests of unconditional
               coverage, with the paper's non-rejection regions and power tables
  acerbi       Acerbi-Szekely (2014) Expected Shortfall backtests Z1, Z2, Z3, with
               Monte Carlo p-values and the Z2 traffic light
  risk         portfolio losses and VaR/CVaR
  predictive   generated windows -> the predictive P&L law the tests need

Driver: `run_backtest.py`.  This package works in P&L space (X < 0 is a loss) as both
papers do; see `predictive` for the mapping from loss space.
"""

from .risk import (portfolio_losses, var_cvar_empirical, var_cvar_gpd,
                   estimate_risk)
from .predictive import PredictivePnL, predictive_from_windows, realized_pnl
from .kupiec import (pf_test, tuff_test, pf_lr, tuff_lr, first_failure,
                     pf_nonrejection_region, tuff_nonrejection_region, pf_power,
                     pf_max_sample_size, reproduce_exhibits)
from .acerbi import (ASResult, test_z1, test_z2, test_z3, z1_statistic,
                     z2_statistic, z3_statistic, es_estimator_mean,
                     traffic_light, Z2_TRAFFIC_LIGHT)

__all__ = [
    "portfolio_losses", "var_cvar_empirical", "var_cvar_gpd", "estimate_risk",
    "PredictivePnL", "predictive_from_windows", "realized_pnl",
    "pf_test", "tuff_test", "pf_lr", "tuff_lr", "first_failure",
    "pf_nonrejection_region", "tuff_nonrejection_region", "pf_power",
    "pf_max_sample_size", "reproduce_exhibits",
    "ASResult", "test_z1", "test_z2", "test_z3", "z1_statistic", "z2_statistic",
    "z3_statistic", "es_estimator_mean", "traffic_light", "Z2_TRAFFIC_LIGHT",
]
