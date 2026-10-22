"""tailfm: EVT marginal PIT -> CFM with a Student-t base -> sampling -> inverse PIT.

The model only.  Scoring lives outside the package (diagnostics.py, figures.py,
backtest/) so tailfm and the baselines are evaluated by identical code.
"""

from .evt import MarginalEnsemble, SemiParametricMarginal, hill_estimator
from .base import sample_base
from .model import VelocityField
from .cfm import train_cfm, sample, EMA
from .data import make_windows

__all__ = ["MarginalEnsemble", "SemiParametricMarginal", "hill_estimator",
           "sample_base", "VelocityField", "train_cfm", "sample", "EMA",
           "make_windows"]
