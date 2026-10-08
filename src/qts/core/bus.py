"""Coada de evenimente a motorului: prioritizată, deterministă, thread-safe (Req 7.1; design).

Cheia de ordonare este `(ts, prioritate_tip, seq)`:

- `ts` este timpul la care sistemul cunoaște evenimentul (UTC): `ts_receipt` pentru
  `MarketEvent` și `ExecutionEvent`, `ts` pentru `TimerEvent` și `CommandEvent`;
- `prioritate_tip` rezolvă egalitățile de timp după tabelul versionat `BUS_PRIORITY_VERSION`:
  `EXECUTION` < `MARKET` < `TIMER` < `COMMAND`. Execuțiile sunt aplicate înaintea datelor de
  piață cu același timp, deci strategia vede portofoliul actualizat;
- `seq` este un contor monoton al cozii (ordinea inserării), deci ordinea este totală și
  reproductibilă pentru aceeași secvență de inserări.

Adaptoarele I/O pot apela `put` din alte fire; numai firul motorului consumă (`get`,
`get_nowait`, `peek_key`). După `close()`, `put` ridică `BusClosedError`, iar consumatorul
golește evenimentele rămase și primește apoi `None`.
"""

from __future__ import annotations

import heapq
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
from typing import Final, Literal

from qts.core.clock import ensure_utc
from qts.core.models import ExecutionEvent, Frozen, MarketEvent, UtcDatetime
from qts.risk.context import KillSwitchScope

__all__ = [
    "BUS_PRIORITY_VERSION",
    "BusClosedError",
    "BusEvent",
    "BusKey",
    "CommandEvent",
    "Envelope",
    "EventBus",
    "EventKind",
    "TimerEvent",
]

BUS_PRIORITY_VERSION: Final = "bus-priority-v1"


class EventKind(IntEnum):
    """Prioritatea la egalitate de timp (valoare mai mică = procesat mai întâi), v1."""

    EXECUTION = 0
    MARKET = 1
    TIMER = 2
    COMMAND = 3


class TimerEvent(Frozen):
    """Eveniment de timp programat (de exemplu expirarea domeniilor DAY ale Kill_Switch)."""

    name: str
    ts: UtcDatetime


CommandName = Literal["kill_switch.activate", "shutdown"]


class CommandEvent(Frozen):
    """Comandă de operator, aplicată în firul motorului."""

    command: CommandName
    actor: str
    ts: UtcDatetime
    scope: KillSwitchScope | None = None
    instrument: str | None = None
    reason_code: str = "OPERATOR_COMMAND"
    detail: str = ""


BusEvent = MarketEvent | ExecutionEvent | TimerEvent | CommandEvent
BusKey = tuple[datetime, int, int]


class BusClosedError(RuntimeError):
    """`put` după închiderea cozii."""


def classify(event: BusEvent) -> tuple[datetime, EventKind]:
    """Timpul de ordonare și tipul unui eveniment."""
    if isinstance(event, ExecutionEvent):
        return event.ts_receipt, EventKind.EXECUTION
    if isinstance(event, MarketEvent):
        return event.ts_receipt, EventKind.MARKET
    if isinstance(event, TimerEvent):
        return event.ts, EventKind.TIMER
    if isinstance(event, CommandEvent):
        return event.ts, EventKind.COMMAND
    raise TypeError(f"tip de eveniment necunoscut: {type(event).__name__}")


@dataclass(frozen=True, slots=True, order=True)
class Envelope:
    """Plic ordonabil; comparația folosește numai cheia."""

    ts: datetime
    kind: EventKind
    seq: int
    event: BusEvent = field(compare=False)

    @property
    def key(self) -> BusKey:
        return (self.ts, int(self.kind), self.seq)


class EventBus:
    """Coadă de priorități thread-safe pentru producători, consumată de un singur fir."""

    def __init__(self) -> None:
        self._heap: list[Envelope] = []
        self._seq = 0
        self._closed = False
        self._cond = threading.Condition(threading.Lock())

    @property
    def priority_version(self) -> str:
        return BUS_PRIORITY_VERSION

    @property
    def closed(self) -> bool:
        return self._closed

    def __len__(self) -> int:
        with self._cond:
            return len(self._heap)

    def put(self, event: BusEvent, *, internal: bool = False) -> Envelope:
        """Adaugă un eveniment. `internal=True` (numai firul motorului) ignoră închiderea,
        ca urmările evenimentelor deja acceptate (de exemplu execuțiile) să fie procesate."""
        ts, kind = classify(event)
        ts = ensure_utc(ts)
        with self._cond:
            if self._closed and not internal:
                raise BusClosedError("coada de evenimente este închisă")
            self._seq += 1
            env = Envelope(ts, kind, self._seq, event)
            heapq.heappush(self._heap, env)
            self._cond.notify()
            return env

    def peek_key(self) -> BusKey | None:
        with self._cond:
            return self._heap[0].key if self._heap else None

    def get_nowait(self) -> Envelope | None:
        with self._cond:
            return heapq.heappop(self._heap) if self._heap else None

    def get(self, timeout: float | None = None) -> Envelope | None:
        """Blochează până la un eveniment; `None` la timeout sau dacă este închisă și goală."""
        with self._cond:
            if not self._heap and not self._closed:
                self._cond.wait_for(lambda: bool(self._heap) or self._closed, timeout)
            return heapq.heappop(self._heap) if self._heap else None

    def drain(self) -> list[Envelope]:
        """Scoate toate evenimentele, în ordinea cheii."""
        with self._cond:
            out = [heapq.heappop(self._heap) for _ in range(len(self._heap))]
            return out

    def close(self) -> None:
        with self._cond:
            self._closed = True
            self._cond.notify_all()
