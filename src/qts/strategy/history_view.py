"""Vedere imuabilă asupra istoricului, fără acces la viitor (Req 7.2).

`HistoryView` primește seria completă de bare pe instrument și un cursor temporal. Expune
numai barele cu `ts_close <= cursor`; orice cerere care ar depăși cursorul ridică
`LookAheadError`, deci look-ahead-ul este imposibil prin construcție. Vederea nu poate fi
modificată: orice atribuire ridică `ImmutableViewError`. Avansarea cursorului produce o
vedere nouă, cu aceleași date subiacente (fără copiere).
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any, NoReturn

from qts.core.clock import ensure_utc
from qts.core.models import Bar


class HistoryViewError(Exception):
    """Bază pentru erorile vederii istorice."""


class LookAheadError(HistoryViewError):
    """Acces la date ulterioare cursorului curent."""


class ImmutableViewError(HistoryViewError, AttributeError):
    """Încercare de a modifica vederea istorică."""


def _validate_series(instrument: str, bars: Sequence[Bar]) -> tuple[Bar, ...]:
    series = tuple(bars)
    for i, bar in enumerate(series):
        if bar.instrument != instrument:
            raise ValueError(f"bara {i} aparține {bar.instrument!r}, nu {instrument!r}")
        if i > 0 and bar.ts_close <= series[i - 1].ts_close:
            raise ValueError(f"barele pentru {instrument!r} nu sunt strict crescătoare (ts_close)")
    return series


class HistoryView:
    """Istoric vizibil până la `cursor` inclusiv, pe instrument.

    Indexarea este relativă la porțiunea vizibilă: `0` este cea mai veche bară vizibilă,
    `-1` este bara curentă (ultima cu `ts_close <= cursor`).
    """

    __slots__ = ("_closes", "_cursor", "_series", "_visible")

    _series: Mapping[str, tuple[Bar, ...]]
    _closes: Mapping[str, tuple[datetime, ...]]
    _cursor: datetime
    _visible: Mapping[str, int]

    def __init__(self, series: Mapping[str, Sequence[Bar]], cursor: datetime) -> None:
        validated = {sym: _validate_series(sym, bars) for sym, bars in series.items()}
        closes = {sym: tuple(b.ts_close for b in bars) for sym, bars in validated.items()}
        self._init(MappingProxyType(validated), MappingProxyType(closes), ensure_utc(cursor))

    def _init(
        self,
        series: Mapping[str, tuple[Bar, ...]],
        closes: Mapping[str, tuple[datetime, ...]],
        cursor: datetime,
    ) -> None:
        visible = {sym: bisect_right(ts, cursor) for sym, ts in closes.items()}
        object.__setattr__(self, "_series", series)
        object.__setattr__(self, "_closes", closes)
        object.__setattr__(self, "_cursor", cursor)
        object.__setattr__(self, "_visible", MappingProxyType(visible))

    # ------------------------------------------------------------------ construcție

    @classmethod
    def from_bars(cls, bars: Iterable[Bar], cursor: datetime) -> HistoryView:
        """Grupează barele pe instrument și le sortează după `ts_close`."""
        grouped: dict[str, list[Bar]] = {}
        for bar in bars:
            grouped.setdefault(bar.instrument, []).append(bar)
        for items in grouped.values():
            items.sort(key=lambda b: b.ts_close)
        return cls(grouped, cursor)

    def advance(self, cursor: datetime) -> HistoryView:
        """Vedere nouă cu alt cursor, reutilizând datele validate (fără copiere)."""
        view = object.__new__(HistoryView)
        view._init(self._series, self._closes, ensure_utc(cursor))
        return view

    def append(self, bar: Bar) -> HistoryView:
        """Vedere nouă cu `bar` adăugată la seria instrumentului și cursorul la `bar.ts_close`.

        Folosită de motor pe măsură ce barele sosesc: se validează numai bara nouă (strict
        după ultima bară a instrumentului), fără revalidarea întregului istoric. Vederea
        curentă rămâne neschimbată.
        """
        series = self._series.get(bar.instrument, ())
        if series and bar.ts_close <= series[-1].ts_close:
            raise ValueError(
                f"barele pentru {bar.instrument!r} nu sunt strict crescătoare (ts_close)"
            )
        new_series = dict(self._series)
        new_series[bar.instrument] = (*series, bar)
        new_closes = dict(self._closes)
        new_closes[bar.instrument] = (*self._closes.get(bar.instrument, ()), bar.ts_close)
        view = object.__new__(HistoryView)
        view._init(MappingProxyType(new_series), MappingProxyType(new_closes), bar.ts_close)
        return view

    # ------------------------------------------------------------------ imuabilitate

    def __setattr__(self, name: str, value: Any) -> NoReturn:
        raise ImmutableViewError("HistoryView este imuabilă")

    def __delattr__(self, name: str) -> NoReturn:
        raise ImmutableViewError("HistoryView este imuabilă")

    # ------------------------------------------------------------------ interogări

    @property
    def cursor(self) -> datetime:
        return self._cursor

    def instruments(self) -> tuple[str, ...]:
        return tuple(sorted(self._series))

    def count(self, instrument: str) -> int:
        """Numărul de bare vizibile pentru instrument."""
        if instrument not in self._visible:
            raise KeyError(f"instrument necunoscut: {instrument!r}")
        return self._visible[instrument]

    def bar(self, instrument: str, index: int = -1) -> Bar:
        """Bară vizibilă după index (negativ = față de bara curentă)."""
        n = self.count(instrument)
        if index >= n:
            raise LookAheadError(
                f"indexul {index} depășește cursorul ({n} bare vizibile pentru {instrument!r})"
            )
        if index < -n:
            raise IndexError(f"indexul {index} precede începutul istoricului {instrument!r}")
        return self._series[instrument][index if index >= 0 else n + index]

    def current(self, instrument: str) -> Bar | None:
        """Ultima bară vizibilă sau `None` dacă nu există."""
        n = self.count(instrument)
        return self._series[instrument][n - 1] if n else None

    def slice(self, instrument: str, start: int, stop: int) -> tuple[Bar, ...]:
        """Bare vizibile `[start, stop)`, cu indici absoluți nenegativi."""
        if start < 0 or stop < start:
            raise IndexError(f"interval invalid [{start}, {stop})")
        n = self.count(instrument)
        if stop > n:
            raise LookAheadError(
                f"capătul {stop} depășește cursorul ({n} bare vizibile pentru {instrument!r})"
            )
        return self._series[instrument][start:stop]

    def last(self, instrument: str, n: int) -> tuple[Bar, ...]:
        """Cel mult ultimele `n` bare vizibile (mai puține dacă istoricul este scurt)."""
        if n < 0:
            raise ValueError("n trebuie să fie >= 0")
        visible = self.count(instrument)
        return self._series[instrument][max(0, visible - n) : visible]

    def closes(self, instrument: str, n: int) -> tuple[Decimal, ...]:
        """Prețurile de închidere ale ultimelor cel mult `n` bare vizibile."""
        return tuple(b.close for b in self.last(instrument, n))

    def until(self, instrument: str, ts: datetime) -> tuple[Bar, ...]:
        """Barele cu `ts_close <= ts`; `ts` ulterior cursorului ridică `LookAheadError`."""
        ts = ensure_utc(ts)
        if ts > self._cursor:
            raise LookAheadError(f"{ts.isoformat()} este după cursor {self._cursor.isoformat()}")
        self.count(instrument)
        return self._series[instrument][: bisect_right(self._closes[instrument], ts)]

    def __getitem__(self, key: tuple[str, int]) -> Bar:
        instrument, index = key
        return self.bar(instrument, index)

    def __repr__(self) -> str:
        return f"HistoryView(cursor={self._cursor.isoformat()}, instruments={self.instruments()})"
