"""Invalidarea subperioadelor de Backtest cu goluri peste prag (Req 7.4, 7.7).

Un *gol* este diferența temporală dintre `ts_close` al unei bare și `ts_open` al barei
următoare a aceluiași instrument. Măsurat în *bare lipsă*:

    missing_bars = round((next.ts_open - prev.ts_close) / interval)

Pentru bare contigue `next.ts_open == prev.ts_close`, deci `missing_bars == 0`. Definiția este
consistentă cu `tests/fixtures/synthetic.Gap`: un `Gap(after_bar, missing_bars)` lasă exact
`missing_bars` sloturi goale, iar între bara dinainte și cea de după golul respectiv timpul
scurs este `missing_bars * interval` minute (vezi `Gap.minutes`).

Politica (`GapPolicy`), per instrument și în ordine temporală strictă (fără look-ahead):

- `missing_bars <= max_gap_bars` (inclusiv golul nul): bara continuă subperioada curentă,
  nimic nu se schimbă (Req 7.7);
- `missing_bars > max_gap_bars`: înaintea barei curente se deschide o subperioadă nouă.
  Subperioada care tocmai s-a încheiat (cea dinaintea golului) este marcată *invalidată*:
  exclusiv ea este afectată (Req 7.4), iar backtestul continuă cu subperioada nouă.

Rezultatul per bară este un `SubperiodMark` cu identificatorul subperioadei, un indicator de
graniță (`boundary`, prima bară a unei subperioade noi deschise de un gol peste prag) și
numărul de bare lipsă față de bara anterioară. O subperioadă este „invalidată” dacă este
urmată de un gol peste prag; `GapPolicy.invalidated_subperiods()` le enumeră după ce fluxul a
fost consumat.

Integrarea cu motorul (`core/engine.py`, seam opțional): la o graniță peste prag motorul
resetează starea strategiei și istoricul vizibil pentru acel instrument, astfel încât niciun
semnal să nu traverseze golul, și suspendă generarea de ordine pentru bara-graniță.
Un gol sub prag nu atinge nimic (Req 7.7). Modulul în sine nu cunoaște motorul: produce numai
marcaje deterministe pe care motorul sau un driver le consultă.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from decimal import Decimal
from typing import Final

from qts.core.clock import ensure_utc
from qts.core.models import Bar, Frozen

__all__ = [
    "GAP_POLICY_VERSION",
    "GapPolicy",
    "SubperiodMark",
    "missing_bars_between",
]

GAP_POLICY_VERSION: Final = "gap-policy-v1"


def missing_bars_between(prev: Bar, nxt: Bar) -> int:
    """Bare lipsă între două bare consecutive ale aceluiași instrument.

    Numărul este `round((next.ts_open - prev.ts_close) / interval)`, folosind intervalul
    (în minute) al barei anterioare. Zero pentru bare contigue. Ridică `ValueError` dacă
    barele nu aparțin aceluiași instrument sau dacă bara următoare nu este ulterioară celei
    anterioare (fluxul trebuie să fie strict crescător).
    """
    if prev.instrument != nxt.instrument:
        raise ValueError(
            f"barele aparțin unor instrumente diferite: {prev.instrument!r} != {nxt.instrument!r}"
        )
    prev_close = ensure_utc(prev.ts_close)
    nxt_open = ensure_utc(nxt.ts_open)
    if nxt_open < prev_close:
        raise ValueError(
            f"bara {nxt_open.isoformat()} precede închiderea anterioară {prev_close.isoformat()}"
        )
    step = timedelta(minutes=prev.interval_min)
    units = (nxt_open - prev_close) / step
    # Rotunjire la cel mai apropiat întreg; micile neregularități de aliniere nu contează.
    return int(Decimal(repr(units)).to_integral_value())


class SubperiodMark(Frozen):
    """Clasificarea unei bare în fluxul unui instrument."""

    instrument: str
    ts_open: datetime
    subperiod_id: int  # 0, 1, 2, ... în ordinea apariției per instrument
    boundary: bool  # True numai pe prima bară a unei subperioade deschise de un gol peste prag
    missing_before: int  # bare lipsă față de bara anterioară (0 pentru prima bară)


class GapPolicy:
    """Partiționează fluxul fiecărui instrument în subperioade pe granițele golurilor mari.

    Starea este per instrument și se actualizează strict în ordinea sosirii barelor
    (`observe`). Nu privește niciodată înainte: decizia pentru o bară folosește numai bara
    anterioară a aceluiași instrument.
    """

    def __init__(self, max_gap_bars: int) -> None:
        if max_gap_bars < 0:
            raise ValueError("max_gap_bars nu poate fi negativ")
        self._max_gap_bars = max_gap_bars
        self._prev: dict[str, Bar] = {}
        self._subperiod: dict[str, int] = {}
        # Subperioade urmate de un gol peste prag (invalidate), per instrument, în ordine.
        self._invalidated: dict[str, list[int]] = {}

    @property
    def max_gap_bars(self) -> int:
        return self._max_gap_bars

    def observe(self, bar: Bar) -> SubperiodMark:
        """Clasifică `bar` și actualizează starea instrumentului.

        Prima bară a unui instrument deschide subperioada 0 (fără graniță). Un gol
        `> max_gap_bars` față de bara anterioară închide și invalidează subperioada curentă și
        deschide una nouă (graniță); un gol `<= max_gap_bars` continuă subperioada (Req 7.7).
        """
        sym = bar.instrument
        prev = self._prev.get(sym)
        if prev is None:
            self._prev[sym] = bar
            self._subperiod[sym] = 0
            return SubperiodMark(
                instrument=sym,
                ts_open=ensure_utc(bar.ts_open),
                subperiod_id=0,
                boundary=False,
                missing_before=0,
            )
        missing = missing_bars_between(prev, bar)
        boundary = missing > self._max_gap_bars
        current = self._subperiod[sym]
        if boundary:
            # Subperioada tocmai încheiată este afectată exclusiv ea (Req 7.4).
            self._invalidated.setdefault(sym, []).append(current)
            current += 1
            self._subperiod[sym] = current
        self._prev[sym] = bar
        return SubperiodMark(
            instrument=sym,
            ts_open=ensure_utc(bar.ts_open),
            subperiod_id=current,
            boundary=boundary,
            missing_before=missing,
        )

    def current_subperiod(self, instrument: str) -> int | None:
        """Subperioada curentă a instrumentului sau `None` dacă nu a sosit nicio bară."""
        return self._subperiod.get(instrument)

    def invalidated_subperiods(self, instrument: str) -> tuple[int, ...]:
        """Subperioadele invalidate (urmate de un gol peste prag) pentru instrument."""
        return tuple(self._invalidated.get(instrument, ()))

    def any_invalidated(self) -> bool:
        """True dacă vreun instrument a avut cel puțin o subperioadă invalidată."""
        return any(self._invalidated.values())
