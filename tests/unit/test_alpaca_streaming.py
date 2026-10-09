"""Teste pentru `AlpacaStreamingSource`: flux continuu determinist (încălzire + sondaje succesive).

Toate testele sunt deterministe și fără rețea: un client Alpaca fals întoarce loturi scriptate
succesive, un ceas fals (`FakeClock`) dă timpul curent, iar `sleep`-ul injectat doar avansează
ceasul fals cu intervalul barei — niciun `time.sleep` real, niciun apel de rețea. Verificăm:
emiterea lotului de încălzire, apoi a barelor nou închise la fiecare sondaj, deduplicarea (nicio
bară reemisă), excluderea barelor invalide, ordinea cronologică și oprirea la seam-ul `max_polls`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from qts.core.models import Bar, MarketEvent
from qts.data.alpaca_source import AlpacaBar, AlpacaStreamingSource

D = Decimal
INTERVAL = 15
DELTA = timedelta(minutes=INTERVAL)
START = datetime(2024, 3, 26, 15, 0, tzinfo=UTC)


class FakeClock:
    """Ceas fals controlat de `sleep`: nu merge înapoi, nu atinge timpul de perete."""

    def __init__(self, start: datetime) -> None:
        self._now = start

    def now(self) -> datetime:
        return self._now

    def sleep(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


class ScriptedClient:
    """Client Alpaca fals: întoarce loturile scriptate în ordine, câte unul per `get_bars`.

    Fiecare apel `get_bars` consumă următorul lot din scenariu; după epuizare întoarce lotul gol.
    Reține apelurile (fereastra [start, end]) pentru inspecție în teste.
    """

    def __init__(self, batches: Sequence[Sequence[AlpacaBar]]) -> None:
        self._batches = [list(b) for b in batches]
        self._i = 0
        self.calls: list[tuple[str, int, datetime, datetime]] = []

    def get_bars(
        self, symbol: str, interval_min: int, start: datetime, end: datetime
    ) -> Sequence[AlpacaBar]:
        self.calls.append((symbol, interval_min, start, end))
        if self._i < len(self._batches):
            batch = self._batches[self._i]
            self._i += 1
            return batch
        return []


def _ab(ts: datetime, close: float = 100.0, *, o: float | None = None, h: float | None = None,
        lo: float | None = None, v: float = 1000.0) -> AlpacaBar:
    """Construiește o bară validă: high/low încadrează mereu open și close (OHLC consistent)."""
    open_ = o if o is not None else close
    high = h if h is not None else max(open_, close) + 1.0
    low = lo if lo is not None else min(open_, close) - 1.0
    return AlpacaBar(ts_open=ts, open=open_, high=high, low=low, close=close, volume=v)


def _source(
    client: ScriptedClient,
    clock: FakeClock,
    *,
    warmup: bool = True,
    max_polls: int | None = None,
    lookback: timedelta = timedelta(hours=1),
) -> AlpacaStreamingSource:
    return AlpacaStreamingSource(
        symbol="SPY",
        interval_min=INTERVAL,
        tick_size=D("0.01"),
        client=client,
        now=clock.now,
        sleep=clock.sleep,
        lookback=lookback,
        warmup=warmup,
        max_polls=max_polls,
    )


def _closes(events: Sequence[MarketEvent]) -> list[datetime]:
    out: list[datetime] = []
    for e in events:
        assert isinstance(e.payload, Bar)
        out.append(e.payload.ts_close)
    return out


# --------------------------------------------------------------------------- încălzire + sondaje


def test_warmup_then_successive_new_bars_across_polls() -> None:
    """Emite lotul de încălzire, apoi barele nou închise la fiecare sondaj (mai mult de unul)."""
    # Lot de încălzire: două bare istorice. Apoi câte o bară nouă la fiecare sondaj.
    warmup = [_ab(START, 100.0), _ab(START + DELTA, 100.5)]
    poll1 = [_ab(START + 2 * DELTA, 101.0)]
    poll2 = [_ab(START + 3 * DELTA, 101.5)]
    poll3 = [_ab(START + 4 * DELTA, 102.0)]
    client = ScriptedClient([warmup, poll1, poll2, poll3])
    clock = FakeClock(START + 2 * DELTA)
    source = _source(client, clock, max_polls=3)

    events = list(source.stream())

    # 2 (încălzire) + 3 (câte o bară pe sondaj) = 5 bare, în ordine cronologică strictă.
    closes = _closes(events)
    assert len(closes) == 5
    assert closes == sorted(closes)
    assert closes == [
        START + DELTA,
        START + 2 * DELTA,
        START + 3 * DELTA,
        START + 4 * DELTA,
        START + 5 * DELTA,
    ]
    # Barele din mai multe sondaje au fost procesate (nu doar un lot): warmup + 3 sondaje.
    assert source.polls == 4


def test_deduplicates_never_reemits_a_bar() -> None:
    """Barele deja emise (după ts_close) nu sunt reemise la sondaje ulterioare (deduplicare)."""
    warmup = [_ab(START, 100.0), _ab(START + DELTA, 100.5)]
    # Sondajul 1 repetă ultima bară de încălzire + una nouă.
    poll1 = [_ab(START + DELTA, 100.5), _ab(START + 2 * DELTA, 101.0)]
    # Sondajul 2 repetă tot ce s-a văzut + una nouă.
    poll2 = [
        _ab(START, 100.0),
        _ab(START + DELTA, 100.5),
        _ab(START + 2 * DELTA, 101.0),
        _ab(START + 3 * DELTA, 101.5),
    ]
    client = ScriptedClient([warmup, poll1, poll2])
    clock = FakeClock(START + 2 * DELTA)
    source = _source(client, clock, max_polls=2)

    closes = _closes(list(source.stream()))

    # Fiecare bară apare exact o dată, în ordine.
    assert closes == [START + DELTA, START + 2 * DELTA, START + 3 * DELTA, START + 4 * DELTA]
    assert len(closes) == len(set(closes))


def test_invalid_bars_excluded_and_recorded() -> None:
    """Barele invalide (OHLC imposibil) sunt excluse și reținute, nu emise către strategie."""
    bad = AlpacaBar(ts_open=START + DELTA, open=100.0, high=98.0, low=99.0, close=100.0, volume=5.0)
    warmup = [_ab(START, 100.0), bad, _ab(START + 2 * DELTA, 101.0)]
    client = ScriptedClient([warmup])
    clock = FakeClock(START + 2 * DELTA)
    source = _source(client, clock, max_polls=0)  # doar încălzirea

    closes = _closes(list(source.stream()))

    assert closes == [START + DELTA, START + 3 * DELTA]  # bara coruptă lipsește
    assert len(source.rejections) == 1


def test_sleep_advances_injected_clock_each_poll() -> None:
    """`sleep` avansează ceasul fals cu exact intervalul barei la fiecare sondaj (determinist)."""
    client = ScriptedClient([[], [_ab(START, 100.0)], [_ab(START + DELTA, 100.5)]])
    clock = FakeClock(START)
    source = _source(client, clock, warmup=False, max_polls=2)

    list(source.stream())

    # Două sondaje → ceasul a avansat cu 2 × interval față de START.
    assert clock.now() == START + 2 * DELTA
    # Ferestrele cerute clientului se mută înainte cu fiecare sondaj.
    assert len(client.calls) == 2


def test_terminates_at_max_polls_seam() -> None:
    """Fluxul se oprește exact după `max_polls` sondaje, deterministic (fără buclă infinită)."""
    # Client care ar putea produce bare la nesfârșit; ne bazăm pe seam-ul de oprire.
    many = [[_ab(START + i * DELTA, 100.0 + i)] for i in range(100)]
    client = ScriptedClient(many)
    clock = FakeClock(START)
    source = _source(client, clock, warmup=False, max_polls=5)

    events = list(source.stream())

    assert source.polls == 5  # fără warmup: exact max_polls sondaje
    assert len(events) == 5


def test_should_continue_predicate_stops_stream() -> None:
    """Predicatul `should_continue` oprește fluxul când întoarce False (seam de oprire)."""
    many = [[_ab(START + i * DELTA, 100.0 + i)] for i in range(100)]
    client = ScriptedClient(many)
    clock = FakeClock(START)
    state = {"n": 0}

    def should_continue() -> bool:
        state["n"] += 1
        return state["n"] <= 3  # permite 3 sondaje, apoi oprește

    source = AlpacaStreamingSource(
        symbol="SPY",
        interval_min=INTERVAL,
        tick_size=D("0.01"),
        client=client,
        now=clock.now,
        sleep=clock.sleep,
        warmup=False,
        should_continue=should_continue,
    )

    events = list(source.stream())
    assert len(events) == 3


def test_warmup_disabled_emits_nothing_before_first_poll() -> None:
    """Fără încălzire, nu se emite niciun lot istoric; primul eveniment vine din primul sondaj."""
    client = ScriptedClient([[_ab(START, 100.0)]])
    clock = FakeClock(START)
    source = _source(client, clock, warmup=False, max_polls=1)

    events = list(source.stream())
    assert len(events) == 1
    # Un singur apel la client (sondajul), fără apel de încălzire.
    assert len(client.calls) == 1


def test_prices_are_decimal_rounded_to_tick() -> None:
    """Conversia float→Decimal cu rotunjire la tick, identică cu adaptorul one-shot."""
    client = ScriptedClient([[_ab(START, 100.123, o=100.126, h=100.129, lo=100.121)]])
    clock = FakeClock(START)
    source = _source(client, clock, max_polls=0)
    event = next(iter(source.stream()))
    assert isinstance(event.payload, Bar)
    assert event.payload.close == D("100.12")
    assert isinstance(event.payload.close, Decimal)


def test_rejects_bad_params() -> None:
    client = ScriptedClient([])
    clock = FakeClock(START)
    with pytest.raises(ValueError, match="interval_min"):
        AlpacaStreamingSource(
            symbol="SPY", interval_min=1, tick_size=D("0.01"),
            client=client, now=clock.now, sleep=clock.sleep,
        )
    with pytest.raises(ValueError, match="tick_size"):
        AlpacaStreamingSource(
            symbol="SPY", interval_min=15, tick_size=D("0"),
            client=client, now=clock.now, sleep=clock.sleep,
        )
    with pytest.raises(ValueError, match="max_polls"):
        AlpacaStreamingSource(
            symbol="SPY", interval_min=15, tick_size=D("0.01"),
            client=client, now=clock.now, sleep=clock.sleep, max_polls=-1,
        )
