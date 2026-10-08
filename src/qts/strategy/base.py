"""Contractul strategiilor: funcții pure, deterministe și explicabile (Req 6.1, 6.2, 6.6).

O strategie primește bara curentă, o `HistoryView` limitată la trecut și starea anterioară
și întoarce un `Signal` împreună cu starea nouă. Nu primește ceas, I/O, aleatorietate sau
acces la portofoliu ori risc și nu stabilește cantitatea (aceasta aparține `Risk_Engine`).
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Protocol, runtime_checkable

from qts.core.models import Bar, Frozen, Signal, SignalAction
from qts.strategy.history_view import HistoryView

# Coduri de motiv standard pentru semnalele fără acțiune (Req 6.4).
REASON_INVALID_INPUT = "INVALID_INPUT"
REASON_INSUFFICIENT_HISTORY = "INSUFFICIENT_HISTORY"
REASON_NO_RULE_MATCHED = "NO_RULE_MATCHED"


class StrategyState(Frozen):
    """Starea unei strategii: imuabilă și serializabilă canonic (`canonical_json`).

    Strategiile concrete o extind cu propriile câmpuri, toate imuabile.
    """


@runtime_checkable
class Strategy(Protocol):
    strategy_id: str
    version: str

    def initial_state(self) -> StrategyState: ...

    def on_bar(
        self, bar: Bar, view: HistoryView, state: StrategyState
    ) -> tuple[Signal, StrategyState]: ...


def bar_data_id(bar: Bar) -> str:
    """Identificatorul stabil al unei bare, folosit în `Signal.data_ids` (Req 6.5)."""
    return f"{bar.instrument}@{bar.ts_close.isoformat()}"


def make_signal_id(strategy_id: str, version: str, bar: Bar) -> str:
    """Identificator determinist: aceeași strategie pe aceeași bară dă același id (Req 6.1)."""
    raw = f"{strategy_id}|{version}|{bar.instrument}|{bar.ts_close.isoformat()}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def build_signal(
    strategy: Strategy,
    bar: Bar,
    *,
    action: SignalAction,
    reason_code: str,
    config_snapshot_id: str,
    inputs: Mapping[str, Decimal] | None = None,
    rules_evaluated: Sequence[str] = (),
    data_ids: Sequence[str] | None = None,
    stop_price: Decimal | None = None,
) -> Signal:
    """Construiește un `Signal` complet explicabil (Req 6.3, 6.5)."""
    return Signal(
        signal_id=make_signal_id(strategy.strategy_id, strategy.version, bar),
        strategy_id=strategy.strategy_id,
        strategy_version=strategy.version,
        instrument=bar.instrument,
        ts=bar.ts_close,
        action=action,
        stop_price=stop_price,
        reason_code=reason_code,
        inputs=dict(inputs or {}),
        rules_evaluated=list(rules_evaluated),
        config_snapshot_id=config_snapshot_id,
        data_ids=list(data_ids) if data_ids is not None else [bar_data_id(bar)],
    )


def none_signal(
    strategy: Strategy,
    bar: Bar,
    *,
    reason_code: str,
    config_snapshot_id: str,
    inputs: Mapping[str, Decimal] | None = None,
    rules_evaluated: Sequence[str] = (),
    data_ids: Sequence[str] | None = None,
) -> Signal:
    """Semnal fără acțiune cu cod de motiv, pentru intrări absente sau invalide (Req 6.4)."""
    return build_signal(
        strategy,
        bar,
        action="NONE",
        reason_code=reason_code,
        config_snapshot_id=config_snapshot_id,
        inputs=inputs,
        rules_evaluated=rules_evaluated,
        data_ids=data_ids,
    )
