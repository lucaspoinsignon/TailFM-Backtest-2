"""Shared utilities for the baselines: scalers matching each reference
implementation's convention, and small torch helpers."""

from __future__ import annotations

import numpy as np
import torch


def default_device(device: str | None = None) -> str:
    return device or ("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class MinMax01Scaler:
    """Per-feature min-max scaling to [0, 1] over all windows and time steps, as in
    both the TimeGAN and TimeVAE references.  Their models end in a sigmoid, so this
    is part of the model rather than a preprocessing choice.
    """

    def fit(self, windows: np.ndarray) -> "MinMax01Scaler":
        self.min_ = windows.min(axis=(0, 1))                    # (f,)
        self.max_ = (windows - self.min_).max(axis=(0, 1))      # (f,) range
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return (windows - self.min_) / (self.max_ + 1e-7)

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.fit(windows).transform(windows)

    def inverse_transform(self, windows01: np.ndarray) -> np.ndarray:
        return windows01 * self.max_ + self.min_


class SymmetricMaxScaler:
    """Per-feature scaling to [-1, 1] by the max absolute value, since the Tail-GAN
    generator hard-clamps its output to that range.  The clamp means Tail-GAN cannot
    generate a return more extreme than the training maximum.
    """

    def fit(self, windows: np.ndarray) -> "SymmetricMaxScaler":
        self.scale_ = np.abs(windows).max(axis=(0, 1)) + 1e-12  # (f,)
        return self

    def transform(self, windows: np.ndarray) -> np.ndarray:
        return windows / self.scale_

    def fit_transform(self, windows: np.ndarray) -> np.ndarray:
        return self.fit(windows).transform(windows)

    def inverse_transform(self, windows_pm1: np.ndarray) -> np.ndarray:
        return windows_pm1 * self.scale_
