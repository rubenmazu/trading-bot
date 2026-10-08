"""Constructori comuni pentru teste."""

from __future__ import annotations

import copy
from typing import Any

BASE_CONFIG: dict[str, Any] = {
    "schema_version": "1",
    "environment": "backtest",
    "run": {"seed": 42, "db_path": "runs/test.db"},
    "data": {
        "source_id": "synthetic",
        "dataset_path": "data/synthetic.csv",
        "bar_interval_min": 15,
        "default_freshness_seconds": 1800,
    },
    "broker": {"kind": "sim"},
    "strategy": {"strategy_id": "mean_reversion_v1"},
    "instruments": [
        {
            "symbol": "XYZ",
            "venue": "XETR",
            "asset_class": "etf",
            "currency": "EUR",
            "tick_size": "0.01",
            "qty_step": "0.001",
            "min_qty": "0.001",
            "calendar_id": "XETR",
            "fractional": True,
        }
    ],
}


def config_dict(**overrides: Any) -> dict[str, Any]:
    """Copie profundă a configurației de bază cu suprascrieri pe secțiuni de prim nivel."""
    data = copy.deepcopy(BASE_CONFIG)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(data.get(key), dict):
            data[key] = {**data[key], **value}
        else:
            data[key] = value
    return data


DEMO_BROKER: dict[str, Any] = {
    "kind": "demo",
    "name": "ibkr",
    "endpoint": "127.0.0.1:7497",
    "account_id": "DU1234567",
    "secret_ref": "qts/demo/broker",
}
