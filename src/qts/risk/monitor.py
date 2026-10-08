"""Monitorizarea continuă a pierderii zilnice și totale (Req 13.4, 13.6, 13.11; design Risk_Engine).

`LossMonitor` primește, la fiecare `Market_Event` și `Execution_Event`, o `LossObservation`
(PnL zilnic realizat și nerealizat, capitalul curent, retragerile și depunerile cumulate, toate
`Decimal` în EUR) și emite comenzi `KillSwitchActivation` către un `KillSwitchSink` injectat:

- pierderea zilnică (realizată + nerealizată) ≥ limita zilnică → `Kill_Switch(scope=DAY)`
  pentru `Trading_Day` curent (13.4). Domeniul DAY expiră la următorul `Trading_Day`;
- `equity + withdrawals - deposits ≤ reference_capital - total_loss_limit` (90 EUR) →
  `Kill_Switch(scope=CAPITAL_CONFIG, permanent=True)` (13.6). Activarea nu expiră și nu poate
  fi anulată de operator; singura cale de reluare este `start_new_capital_config` cu o
  `CapitalChangeAuthorization` validă (13.11, 28.4).

Activările sunt idempotente: cel mult una pe (DAY, Trading_Day) și una pe configurație de capital.
Starea se marchează activă înaintea emiterii (fail-closed: Risk_Engine blochează chiar dacă
sink-ul eșuează); activările neconfirmate de sink rămân în `pending` și se reemit la următoarea
observație sau la `flush()`. Fiecare activare are un `activation_id` determinist, ca
destinatarul să poată deduplica.

Starea este un model serializabil (`export_state` / parametrul `state`), astfel încât activarea
permanentă supraviețuiește repornirii. Persistarea efectivă în jurnal și componenta
`Kill_Switch` sunt în afara acestui modul (sarcinile 11.3, 13.2).

`Trading_Day` se obține printr-o funcție injectabilă `TradingDayFn`; implicit este data UTC.
`zoned_trading_day` acceptă orice `tzinfo` (de exemplu `ZoneInfo` din calendarul pieței).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, time, timedelta, tzinfo
from decimal import Decimal
from enum import StrEnum
from typing import Protocol, runtime_checkable

from pydantic import model_validator

from qts.config.schema import RiskConfig
from qts.core.clock import ensure_utc
from qts.core.models import Dec, Frozen, UtcDatetime
from qts.core.money import ZERO

from .context import KillSwitchScope, KillSwitchState

__all__ = [
    "CapitalChangeAuthorization",
    "CapitalConfigChangeError",
    "KillSwitchActivation",
    "KillSwitchSink",
    "LossMonitor",
    "LossMonitorState",
    "LossObservation",
    "LossTriggerReason",
    "TradingDayFn",
    "utc_trading_day",
    "zoned_trading_day",
]

TradingDayFn = Callable[[datetime], date]


def utc_trading_day(ts: datetime) -> date:
    """`Trading_Day` implicit: data calendaristică UTC a evenimentului."""
    return ensure_utc(ts).date()


def zoned_trading_day(tz: tzinfo, rollover: time = time(0)) -> TradingDayFn:
    """`Trading_Day` în fusul pieței, cu trecerea la ziua următoare la ora `rollover` locală."""
    shift = timedelta(hours=rollover.hour, minutes=rollover.minute, seconds=rollover.second)

    def _day(ts: datetime) -> date:
        return (ensure_utc(ts).astimezone(tz) - shift).date()

    return _day


class LossTriggerReason(StrEnum):
    DAILY_LOSS_LIMIT_REACHED = "LOSS_DAILY_LIMIT_REACHED"
    TOTAL_LOSS_LIMIT_REACHED = "LOSS_TOTAL_LIMIT_REACHED"


class LossObservation(Frozen):
    """Instantaneul contabil după un `Market_Event` sau `Execution_Event` (EUR)."""

    ts: UtcDatetime
    daily_realized_pnl_eur: Dec  # PnL net realizat în Trading_Day al lui `ts`
    daily_unrealized_pnl_eur: Dec
    equity_eur: Dec  # capitalul curent: numerar + poziții marcate
    cumulative_withdrawals_eur: Dec = ZERO  # în configurația de capital curentă
    cumulative_deposits_eur: Dec = ZERO  # depunerile ulterioare configurării

    @model_validator(mode="after")
    def _check(self) -> LossObservation:
        if self.cumulative_withdrawals_eur < 0 or self.cumulative_deposits_eur < 0:
            raise ValueError("retragerile și depunerile cumulate nu pot fi negative")
        return self

    @property
    def daily_loss_eur(self) -> Decimal:
        return max(ZERO, -(self.daily_realized_pnl_eur + self.daily_unrealized_pnl_eur))

    @property
    def adjusted_capital_eur(self) -> Decimal:
        """Capital curent + retrageri cumulate − depuneri ulterioare (Req 13.6)."""
        return self.equity_eur + self.cumulative_withdrawals_eur - self.cumulative_deposits_eur


class KillSwitchActivation(Frozen):
    """Comanda de activare emisă către componenta `Kill_Switch`."""

    activation_id: str
    scope: KillSwitchScope
    reason: LossTriggerReason
    value: Dec  # pierderea zilnică sau capitalul ajustat calculat
    limit: Dec  # limita zilnică sau pragul de capital (90 EUR)
    ts: UtcDatetime
    permanent: bool
    capital_config_id: str
    trading_day: date | None = None  # numai pentru DAY


@runtime_checkable
class KillSwitchSink(Protocol):
    """Destinatarul activărilor; trebuie să tolereze reemiterea aceluiași `activation_id`."""

    def activate(self, activation: KillSwitchActivation) -> None: ...


@runtime_checkable
class CapitalChangeAuthorization(Protocol):
    """Loc rezervat pentru `Capital_Change_Authorization` (Req 28, sarcina 17.4).

    Monitorul nu validează conținutul autorizării: acceptă numai un obiect deja validat de
    componenta dedicată, care declară explicit validitatea și noua configurație de capital.
    """

    @property
    def new_capital_config_id(self) -> str: ...

    def is_valid(self) -> bool: ...


class CapitalConfigChangeError(Exception):
    """Schimbarea configurației de capital a fost refuzată."""


class LossMonitorState(Frozen):
    """Stare serializabilă a monitorului, pentru persistare și repornire."""

    capital_config_id: str
    capital_config_activation: KillSwitchActivation | None = None
    day_activation: KillSwitchActivation | None = None
    current_trading_day: date | None = None
    pending: tuple[KillSwitchActivation, ...] = ()
    retired_capital_config_ids: frozenset[str] = frozenset()

    @model_validator(mode="after")
    def _check(self) -> LossMonitorState:
        if not self.capital_config_id.strip():
            raise ValueError("capital_config_id este obligatoriu")
        if self.capital_config_id in self.retired_capital_config_ids:
            raise ValueError("configurația de capital curentă a fost retrasă")
        act = self.capital_config_activation
        if act is not None and (
            act.scope is not KillSwitchScope.CAPITAL_CONFIG
            or not act.permanent
            or act.capital_config_id != self.capital_config_id
        ):
            raise ValueError("activarea CAPITAL_CONFIG nu corespunde configurației curente")
        if self.day_activation is not None and self.day_activation.scope is not KillSwitchScope.DAY:
            raise ValueError("day_activation trebuie să aibă domeniul DAY")
        return self

    @property
    def capital_config_active(self) -> bool:
        return self.capital_config_activation is not None


class LossMonitor:
    """Evaluează fiecare observație și emite activări `Kill_Switch` idempotente."""

    def __init__(
        self,
        config: RiskConfig,
        sink: KillSwitchSink,
        *,
        capital_config_id: str,
        trading_day: TradingDayFn = utc_trading_day,
        state: LossMonitorState | None = None,
    ) -> None:
        if state is not None and state.capital_config_id != capital_config_id:
            raise ValueError(
                f"starea restaurată aparține configurației {state.capital_config_id}, "
                f"nu {capital_config_id}"
            )
        self._daily_limit = config.daily_loss_limit_eur
        self._capital_floor = config.reference_capital_eur - config.total_loss_limit_eur
        self._sink = sink
        self._trading_day = trading_day
        self._state = state or LossMonitorState(capital_config_id=capital_config_id)

    # ------------------------------------------------------------------ interogări

    @property
    def capital_floor_eur(self) -> Decimal:
        return self._capital_floor

    @property
    def daily_limit_eur(self) -> Decimal:
        return self._daily_limit

    def export_state(self) -> LossMonitorState:
        return self._state

    def is_day_active(self, ts: datetime) -> bool:
        act = self._state.day_activation
        return act is not None and act.trading_day == self._trading_day(ts)

    def kill_switch_state(
        self, ts: datetime, base: KillSwitchState | None = None
    ) -> KillSwitchState:
        """Starea pentru Risk_Engine la `ts`, combinată (OR) cu domeniile din `base`."""
        base = base or KillSwitchState()
        return base.model_copy(
            update={
                "day_active": base.day_active or self.is_day_active(ts),
                "capital_config_active": base.capital_config_active
                or self._state.capital_config_active,
            }
        )

    # ------------------------------------------------------------------ evaluare

    def observe(self, obs: LossObservation) -> tuple[KillSwitchActivation, ...]:
        """Evaluează observația; întoarce activările noi (deja marcate în stare)."""
        st = self._state
        day = self._trading_day(obs.ts)
        new: list[KillSwitchActivation] = []
        updates: dict[str, object] = {}

        if st.current_trading_day is None or day > st.current_trading_day:
            updates["current_trading_day"] = day
            current_day = day
        else:
            current_day = st.current_trading_day

        if not st.capital_config_active and obs.adjusted_capital_eur <= self._capital_floor:
            act = KillSwitchActivation(
                activation_id=f"{KillSwitchScope.CAPITAL_CONFIG}:{st.capital_config_id}",
                scope=KillSwitchScope.CAPITAL_CONFIG,
                reason=LossTriggerReason.TOTAL_LOSS_LIMIT_REACHED,
                value=obs.adjusted_capital_eur,
                limit=self._capital_floor,
                ts=obs.ts,
                permanent=True,
                capital_config_id=st.capital_config_id,
            )
            updates["capital_config_activation"] = act
            new.append(act)

        # Observațiile întârziate dintr-un Trading_Day trecut nu mai activează domeniul DAY.
        day_already = st.day_activation is not None and st.day_activation.trading_day == day
        if day == current_day and not day_already and obs.daily_loss_eur >= self._daily_limit:
            act = KillSwitchActivation(
                activation_id=f"{KillSwitchScope.DAY}:{st.capital_config_id}:{day.isoformat()}",
                scope=KillSwitchScope.DAY,
                reason=LossTriggerReason.DAILY_LOSS_LIMIT_REACHED,
                value=obs.daily_loss_eur,
                limit=self._daily_limit,
                ts=obs.ts,
                permanent=False,
                capital_config_id=st.capital_config_id,
                trading_day=day,
            )
            updates["day_activation"] = act
            new.append(act)

        if new:
            updates["pending"] = (*st.pending, *new)
        if updates:
            self._state = st.model_copy(update=updates)
        self.flush()
        return tuple(new)

    def flush(self) -> None:
        """Trimite activările în așteptare; o eroare a sink-ului le păstrează pentru reemitere."""
        while self._state.pending:
            head = self._state.pending[0]
            self._sink.activate(head)
            self._state = self._state.model_copy(update={"pending": self._state.pending[1:]})

    # ------------------------------------------------------------------ configurație nouă

    def start_new_capital_config(self, authorization: CapitalChangeAuthorization) -> None:
        """Singura cale de reluare după atingerea limitei totale (Req 13.11, 28.4).

        Necesită o autorizare validă pentru o configurație nouă, niciodată folosită. Domeniul
        DAY al zilei curente rămâne activ (fail-closed): limita zilnică ține de Trading_Day.
        Activările în așteptare se trimit înainte de schimbare.
        """
        if not isinstance(authorization, CapitalChangeAuthorization):
            raise CapitalConfigChangeError("autorizarea nu respectă protocolul cerut")
        if not authorization.is_valid():
            raise CapitalConfigChangeError("Capital_Change_Authorization invalidă")
        st = self._state
        new_id = authorization.new_capital_config_id
        if not new_id.strip():
            raise CapitalConfigChangeError("noua configurație de capital nu are identificator")
        if new_id == st.capital_config_id or new_id in st.retired_capital_config_ids:
            raise CapitalConfigChangeError(
                f"configurația {new_id} nu este nouă; Kill_Switch nu se poate dezactiva "
                "în configurația existentă"
            )
        self.flush()
        self._state = LossMonitorState(
            capital_config_id=new_id,
            day_activation=st.day_activation,
            current_trading_day=st.current_trading_day,
            retired_capital_config_ids=st.retired_capital_config_ids | {st.capital_config_id},
        )
