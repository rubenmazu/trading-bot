"""Adaptor de date Alpaca pentru modul Shadow: bare OHLCV reale → `MarketEvent` (Req 5.1, 5.3).

`AlpacaDataAdapter` implementează portul `DataAdapter` folosind datele istorice/curente de la
Alpaca (acțiuni și ETF-uri US, cash). Este folosit în modul **Shadow**: feed real, dar execuție
simulată local prin `SimBroker` — nu pleacă niciun ordin real (Req 1.1, 16.1). Nu există cale
către Live în acest modul.

Principii respectate din restul sistemului:

- **Fără float pe calea banilor**: barele Alpaca sosesc ca `float` (OHLCV). Le convertim în
  `Decimal` prin `str(float)` (reprezentarea text, fără eroare binară) și le rotunjim la
  `tick_size`-ul instrumentului, exact ca generatorul sintetic.
- **Timp UTC**: Alpaca întoarce timestamp-uri UTC pentru fiecare bară; `ts_close = ts_open +
  interval`. În feed, `ts_receipt = ts_close` (bara devine disponibilă la închidere), la fel ca
  reluarea istorică din `CsvSource`.
- **Validare**: fiecare bară trece prin `BarValidator`; barele invalide sunt excluse și motivul
  reținut în `rejections`, nu propagate în strategie (Req 5.4).
- **Secrete**: cheile API vin din `Secret_Store` (keyring) prin `SecretRef`, niciodată din config
  sau cod (Req 23). Clientul concret este injectabil, deci testele rulează fără rețea și fără chei.

Clientul Alpaca este abstractizat prin protocolul `AlpacaBarClient`, cu o singură metodă
`get_bars`. Implementarea reală (`AlpacaPyClient`) folosește pachetul `alpaca-py` și este
construită doar la rulare; testele injectează un client fals determinist.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Final, Protocol, runtime_checkable

from qts.core.models import Bar, MarketEvent
from qts.core.money import dec, round_to_tick
from qts.data.validate import BarValidator
from qts.secrets.store import Identity, SecretRef, SecretStore

logger = logging.getLogger(__name__)

__all__ = [
    "AlpacaBar",
    "AlpacaBarClient",
    "AlpacaClientFactory",
    "AlpacaCredentials",
    "AlpacaDataAdapter",
    "AlpacaStreamingSource",
    "BarRejection",
    "NowFn",
    "SleepFn",
    "wall_clock_sleep",
]


@dataclass(frozen=True, slots=True)
class AlpacaBar:
    """O bară OHLCV brută, așa cum o întoarce clientul Alpaca (preț/volum ca `float`).

    `ts_open` este timpul de deschidere al barei (UTC). Conversia în `Decimal` și validarea se
    fac în adaptor, nu aici: acest obiect este doar transportul neutru dintre client și adaptor.
    """

    ts_open: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@runtime_checkable
class AlpacaBarClient(Protocol):
    """Clientul care aduce bare de la Alpaca. Abstractizat pentru teste fără rețea."""

    def get_bars(
        self, symbol: str, interval_min: int, start: datetime, end: datetime
    ) -> Sequence[AlpacaBar]: ...


@dataclass(frozen=True, slots=True)
class BarRejection:
    """O bară exclusă la validare; `index` este poziția în secvența primită de la client."""

    index: int
    reason: str
    detail: str


@dataclass(frozen=True, slots=True)
class AlpacaCredentials:
    """Referințele (nu valorile) cheilor API Alpaca, rezolvate din Secret_Store (Req 23)."""

    api_key_ref: SecretRef
    api_secret_ref: SecretRef


# Fabrica de client: construiește un AlpacaBarClient din cheile deja dezvăluite. Definit aici,
# înaintea adaptorului, ca adnotarea `from_secret_store` să nu aibă nevoie de ghilimele.
AlpacaClientFactory = Callable[[str, str], AlpacaBarClient]


def _to_decimal(value: float) -> Decimal:
    """Convertește un `float` Alpaca în `Decimal` prin reprezentarea text (fără eroare binară)."""
    return dec(str(value))


class AlpacaDataAdapter:
    """`DataAdapter` pentru Shadow: bare reale Alpaca pentru un singur instrument.

    `source_id` codifică sursa și simbolul, astfel încât snapshot-ul și jurnalul să identifice
    exact de unde vin datele. `tick_size` și `interval_min` corespund instrumentului configurat.

    Clientul (`client`) este injectat: la rulare se dă un `AlpacaPyClient` construit cu cheile din
    Secret_Store; în teste se dă un client fals determinist. Fereastra de timp (`lookback`) spune
    cât istoric recent se cere la pornirea Shadow; feed-ul emite barele în ordine cronologică.
    """

    def __init__(
        self,
        *,
        symbol: str,
        interval_min: int,
        tick_size: Decimal,
        client: AlpacaBarClient,
        clock_now: datetime,
        lookback: timedelta = timedelta(days=5),
    ) -> None:
        if interval_min < 5 or interval_min > 60:
            raise ValueError("interval_min trebuie să fie în [5, 60] (Req 4.5)")
        if tick_size <= 0:
            raise ValueError("tick_size trebuie să fie > 0")
        self.symbol = symbol
        self.interval_min = interval_min
        self.tick_size = tick_size
        self.source_id = f"alpaca:{symbol}"
        self._client = client
        self._now = clock_now if clock_now.tzinfo is not None else clock_now.replace(tzinfo=UTC)
        self._lookback = lookback
        self.rejections: list[BarRejection] = []

    @classmethod
    def from_secret_store(
        cls,
        *,
        symbol: str,
        interval_min: int,
        tick_size: Decimal,
        store: SecretStore,
        credentials: AlpacaCredentials,
        requester: Identity,
        environment: str,
        clock_now: datetime,
        lookback: timedelta = timedelta(days=5),
        client_factory: AlpacaClientFactory | None = None,
    ) -> AlpacaDataAdapter:
        """Construiește adaptorul rezolvând cheile din Secret_Store (Req 23.1, 23.4).

        `client_factory` creează clientul concret din cele două chei dezvăluite; implicit este
        `_default_client_factory`, care importă `alpaca-py` doar la rulare. Testele pot injecta o
        fabrică falsă, deci nu depind nici de rețea, nici de pachetul extern.
        """
        api_key = store.get(credentials.api_key_ref, requester, environment)
        api_secret = store.get(credentials.api_secret_ref, requester, environment)
        factory = client_factory if client_factory is not None else _default_client_factory
        client = factory(api_key.reveal(), api_secret.reveal())
        return cls(
            symbol=symbol,
            interval_min=interval_min,
            tick_size=tick_size,
            client=client,
            clock_now=clock_now,
            lookback=lookback,
        )

    def stream(self) -> Iterator[MarketEvent]:
        """Emite barele recente în ordine cronologică, validate și convertite în `Decimal`."""
        return iter(self.load())

    def load(self) -> list[MarketEvent]:
        start = self._now - self._lookback
        raw = self._client.get_bars(self.symbol, self.interval_min, start, self._now)
        self.rejections = []
        validator = BarValidator()
        delta = timedelta(minutes=self.interval_min)
        events: list[MarketEvent] = []
        for index, ab in enumerate(sorted(raw, key=lambda b: b.ts_open)):
            bar = self._to_bar(ab, delta)
            if bar is None:
                continue
            verdict = validator.validate(bar)
            if verdict.bar is None:
                self._reject(index, str(verdict.reason), verdict.detail)
                continue
            events.append(
                MarketEvent(
                    source_id=self.source_id,
                    instrument=self.symbol,
                    ts_source=bar.ts_close,
                    ts_receipt=bar.ts_close,
                    seq=index,
                    kind="bar",
                    payload=bar,
                )
            )
        logger.info(
            "feed Alpaca source_id=%s accepted=%d rejected=%d",
            self.source_id,
            len(events),
            len(self.rejections),
        )
        return events

    # ------------------------------------------------------------------ intern

    def _to_bar(self, ab: AlpacaBar, delta: timedelta) -> Bar | None:
        ts_open = ab.ts_open if ab.ts_open.tzinfo is not None else ab.ts_open.replace(tzinfo=UTC)
        ts_open = ts_open.astimezone(UTC)
        try:
            return Bar(
                instrument=self.symbol,
                ts_open=ts_open,
                ts_close=ts_open + delta,
                interval_min=self.interval_min,
                open=round_to_tick(_to_decimal(ab.open), self.tick_size),
                high=round_to_tick(_to_decimal(ab.high), self.tick_size),
                low=round_to_tick(_to_decimal(ab.low), self.tick_size),
                close=round_to_tick(_to_decimal(ab.close), self.tick_size),
                volume=_to_decimal(ab.volume),
            )
        except (ValueError, ArithmeticError) as exc:
            self._reject(0, "ALPACA_BAR_INVALID", str(exc))
            return None

    def _reject(self, index: int, reason: str, detail: str) -> None:
        logger.warning(
            "bară Alpaca exclusă source_id=%s index=%d reason=%s detail=%s",
            self.source_id,
            index,
            reason,
            detail,
        )
        self.rejections.append(BarRejection(index, reason, detail))


# Seam-uri de timp pentru fluxul continuu: sursa timpului curent și un „somn" injectabil. În
# producție `now` este `WallClock.now` și `sleep` este `wall_clock_sleep` (singurul `time.sleep`
# real); în teste ambele sunt înlocuite cu funcții deterministe peste un ceas fals, deci nu există
# nici timp de perete, nici rețea.
NowFn = Callable[[], datetime]
SleepFn = Callable[[float], None]


def wall_clock_sleep(seconds: float) -> None:  # pragma: no cover - singurul somn real (wall-clock)
    """Somnul real pe ceasul de perete; izolat aici ca testele să nu aștepte niciodată."""
    import time

    if seconds > 0:
        time.sleep(seconds)


class AlpacaStreamingSource:
    """`DataAdapter` continuu pentru feed-ul Alpaca: bare în timp real, pe măsură ce se închid.

    Spre deosebire de `AlpacaDataAdapter` (o singură tragere de `lookback` la pornire, apoi
    oprire), această sursă rămâne activă: după un lot opțional de încălzire (`lookback`), așteaptă
    până la granița următoarei bare (`interval_min`), interoghează `get_bars` pentru barele mai noi
    decât ultima emisă, le deduplică (după `ts_close`), le validează și le emite în ordine
    cronologică — repetat, până când seam-ul de oprire o cere. Astfel motorul (care iterează peste
    `stream()`) rulează continuu, reacționând la fiecare bară nou închisă, exact cum ar face Live.

    Determinism (fără timp de perete, fără rețea în teste): toate dependențele de timp sunt
    injectate — `now` (sursa timpului curent), `sleep` (așteptarea până la granița barei) și
    `should_continue` / `max_polls` (seam-ul de oprire). Testele dau un ceas fals, un `sleep` care
    doar avansează ceasul fals și un client fals cu loturi succesive scriptate; nimic nu atinge
    rețeaua sau `time.sleep`. În producție `now`/`sleep` provin din `WallClock`, iar singurul
    `time.sleep` real este în `wall_clock_sleep` (marcat `# pragma: no cover`).

    Semantica este identică cu `AlpacaDataAdapter` pe calea barelor (aceeași conversie
    float→Decimal prin `str`, aceeași rotunjire la `tick_size`, aceiași timpi UTC, aceeași
    validare): singura diferență este că fluxul este continuu, nu un singur lot istoric. Această
    sursă este neutră față de mod — Demo o folosește azi, iar Live o va reutiliza identic.
    """

    def __init__(
        self,
        *,
        symbol: str,
        interval_min: int,
        tick_size: Decimal,
        client: AlpacaBarClient,
        now: NowFn,
        sleep: SleepFn = wall_clock_sleep,
        lookback: timedelta = timedelta(days=5),
        warmup: bool = True,
        should_continue: Callable[[], bool] | None = None,
        max_polls: int | None = None,
    ) -> None:
        if interval_min < 5 or interval_min > 60:
            raise ValueError("interval_min trebuie să fie în [5, 60] (Req 4.5)")
        if tick_size <= 0:
            raise ValueError("tick_size trebuie să fie > 0")
        if max_polls is not None and max_polls < 0:
            raise ValueError("max_polls trebuie să fie >= 0")
        self.symbol = symbol
        self.interval_min = interval_min
        self.tick_size = tick_size
        self.source_id = f"alpaca:{symbol}"
        self._client = client
        self._now = now
        self._sleep = sleep
        self._lookback = lookback
        self._warmup = warmup
        self._should_continue = should_continue
        self._max_polls = max_polls
        self._delta = timedelta(minutes=interval_min)
        self._validator = BarValidator()
        # Închiderea ultimei bare emise: granița de deduplicare (nimic la fel sau mai vechi).
        self._last_emitted: datetime | None = None
        self.rejections: list[BarRejection] = []
        self.polls = 0

    def stream(self) -> Iterator[MarketEvent]:
        """Emite barele de încălzire, apoi barele nou închise la fiecare sondaj, până la oprire."""
        now = self._as_utc(self._now())
        if self._warmup:
            yield from self._poll(start=now - self._lookback, end=now)
        poll = 0
        while self._keep_going(poll):
            # Așteaptă până la granița următoarei bare, apoi reinterogează timpul curent.
            self._sleep(float(self._delta.total_seconds()))
            now = self._as_utc(self._now())
            start = self._last_emitted if self._last_emitted is not None else now - self._delta
            yield from self._poll(start=start, end=now)
            poll += 1

    # ------------------------------------------------------------------ intern

    def _keep_going(self, poll: int) -> bool:
        if self._max_polls is not None and poll >= self._max_polls:
            return False
        return self._should_continue is None or self._should_continue()

    def _poll(self, *, start: datetime, end: datetime) -> Iterator[MarketEvent]:
        """Un sondaj: trage barele din [start, end], le deduplică, validează și emite în ordine."""
        self.polls += 1
        raw = self._client.get_bars(self.symbol, self.interval_min, start, end)
        seq = 0
        for index, ab in enumerate(sorted(raw, key=lambda b: b.ts_open)):
            bar = self._to_bar(ab)
            if bar is None:
                continue
            # Deduplicare: nu reemite bare la fel sau mai vechi decât ultima deja emisă.
            if self._last_emitted is not None and bar.ts_close <= self._last_emitted:
                continue
            verdict = self._validator.validate(bar)
            if verdict.bar is None:
                self._reject(index, str(verdict.reason), verdict.detail)
                continue
            self._last_emitted = bar.ts_close
            yield MarketEvent(
                source_id=self.source_id,
                instrument=self.symbol,
                ts_source=bar.ts_close,
                ts_receipt=bar.ts_close,
                seq=seq,
                kind="bar",
                payload=bar,
            )
            seq += 1

    def _as_utc(self, ts: datetime) -> datetime:
        return ts if ts.tzinfo is not None else ts.replace(tzinfo=UTC)

    def _to_bar(self, ab: AlpacaBar) -> Bar | None:
        ts_open = ab.ts_open if ab.ts_open.tzinfo is not None else ab.ts_open.replace(tzinfo=UTC)
        ts_open = ts_open.astimezone(UTC)
        try:
            return Bar(
                instrument=self.symbol,
                ts_open=ts_open,
                ts_close=ts_open + self._delta,
                interval_min=self.interval_min,
                open=round_to_tick(_to_decimal(ab.open), self.tick_size),
                high=round_to_tick(_to_decimal(ab.high), self.tick_size),
                low=round_to_tick(_to_decimal(ab.low), self.tick_size),
                close=round_to_tick(_to_decimal(ab.close), self.tick_size),
                volume=_to_decimal(ab.volume),
            )
        except (ValueError, ArithmeticError) as exc:
            self._reject(0, "ALPACA_BAR_INVALID", str(exc))
            return None

    def _reject(self, index: int, reason: str, detail: str) -> None:
        logger.warning(
            "bară Alpaca exclusă (stream) source_id=%s index=%d reason=%s detail=%s",
            self.source_id,
            index,
            reason,
            detail,
        )
        self.rejections.append(BarRejection(index, reason, detail))


def _default_client_factory(api_key: str, api_secret: str) -> AlpacaBarClient:  # pragma: no cover
    """Construiește clientul real `AlpacaPyClient` (necesită pachetul `alpaca-py` instalat)."""
    return AlpacaPyClient(api_key, api_secret)


# Feed-ul implicit de date: IEX este gratuit și disponibil pe conturile paper/gratuite. SIP
# (datele consolidate de la toate bursele) necesită un abonament plătit la Alpaca; dacă îl ceri
# fără abonament, API-ul răspunde „subscription does not permit querying recent SIP data". Pentru
# Demo/paper folosim IEX. Dacă vreodată ai abonament SIP, setează `feed="sip"`.
DEFAULT_ALPACA_FEED: Final = "iex"


class AlpacaPyClient:  # pragma: no cover - necesită rețea și pachetul extern
    """Client real peste `alpaca-py`. Construit doar la rulare, nu în teste.

    Mapează intervalul în minute pe un `TimeFrame` Alpaca și cere barele istorice/recente pentru
    simbol. Importul pachetului este amânat în constructor, ca modulul `qts.data.alpaca_source`
    să poată fi importat (și testat) fără `alpaca-py` instalat.

    `feed` alege sursa de date Alpaca: implicit `iex` (gratuit, pentru conturi paper). `sip`
    (datele consolidate) cere un abonament plătit; fără el, Alpaca refuză cererea.
    """

    def __init__(self, api_key: str, api_secret: str, *, feed: str = DEFAULT_ALPACA_FEED) -> None:
        from alpaca.data.historical import StockHistoricalDataClient

        self._client = StockHistoricalDataClient(api_key, api_secret)
        self._feed = feed

    def get_bars(
        self, symbol: str, interval_min: int, start: datetime, end: datetime
    ) -> Sequence[AlpacaBar]:
        from alpaca.data.enums import DataFeed
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        timeframe = TimeFrame(amount=interval_min, unit=TimeFrameUnit.Minute)
        request = StockBarsRequest(
            symbol_or_symbols=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            feed=DataFeed(self._feed),
        )
        response = self._client.get_stock_bars(request)
        rows = response.data.get(symbol, [])
        return [
            AlpacaBar(
                ts_open=row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
            for row in rows
        ]
