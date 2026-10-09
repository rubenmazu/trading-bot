"""Portul `DataAdapter`: singura cale prin care datele de piață intră în sistem (Req 5.1)."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from qts.core.models import MarketEvent


@runtime_checkable
class DataAdapter(Protocol):
    source_id: str

    def stream(self) -> Iterator[MarketEvent]: ...  # Backtest: istoric; altfel: curent
