from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import MinMaxScaler
from .fis import TSKFIS


def load_model_npz(path: str | Path) -> tuple[TSKFIS, MinMaxScaler]:
    data = np.load(path)
    model = TSKFIS(
        centers=data["centers"],
        sigmas=data["sigmas"],
        consequents=data["consequents"],
    )
    scaler = MinMaxScaler(
        data_min=float(data["scaler_data_min"]),
        data_max=float(data["scaler_data_max"]),
        out_min=float(data["scaler_out_min"]),
        out_max=float(data["scaler_out_max"]),
    )
    return model, scaler
