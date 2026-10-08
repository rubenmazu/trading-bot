"""Kill_Switch: domenii, persistență write-ahead, politica ordinelor deschise (Req 14, 13.11).

Domenii (`KillSwitchScope`): `INSTRUMENT`, `DAY`, `GLOBAL`, `CAPITAL_CONFIG`. Un ordin nou trece
numai dacă niciun domeniu aplicabil instrumentului nu este activ (design: Kill_Switch).

Blocarea (Req 14.1, 14.2: ≤ 1 s) este sincronă și nu depinde de coada de evenimente:
- `GLOBAL` și `CAPITAL_CONFIG` setează un `threading.Event`, citit fără lock;
- activările sunt publicate ca un tuplu imuabil, înlocuit atomic sub lock; cititorii (de exemplu
  `Fail_Safe_Block`, imediat înaintea fiecărei transmiteri) citesc referința fără să aștepte.
Blocarea în memorie are loc înaintea scrierii în jurnal (fail-closed). Efectele externe (comenzile
de anulare din politica `cancel`) se emit numai după ce activarea a fost persistată în jurnal. Dacă
scrierea eșuează, blocarea rămâne, activarea intră în `unpersisted` și se reîncearcă la `flush()`
sau la reemiterea aceluiași `activation_id`.

Persistență: fiecare schimbare este un `KillSwitchEvent` scris printr-un `KillSwitchStore`
(implicit `JournalKillSwitchStore`, peste jurnalul append-only). La repornire, `KillSwitch`
reconstruiește starea din evenimente, deci un kill switch activ rămâne activ (Req 26.1).

Politica ordinelor deschise (`KillSwitchConfig.open_orders_policy`, implicit `keep`, Req 14.3,
14.7): `keep` nu modifică ordinele; `cancel` produce câte un `CancelCommand` pentru fiecare ordin
deschis din domeniul activat. Kill_Switch nu apelează brokerul: comenzile sunt întoarse în
`ActivationOutcome` și puse în `drain_cancel_commands()` pentru Order_Management_Subsystem.
Recepția execuțiilor, reconcilierea, anulările și auditul nu sunt blocate (Req 14.4).

Reluarea (Req 14.5, 14.6, 14.8) cere, pentru fiecare activare: reconciliere reușită, aprobarea
explicită a operatorului, motivul activării rezolvat, cauza și corecția înregistrate. Reconcilierea
eșuată refuză reluarea indiferent de aprobare. Refuzurile sunt și ele auditate.
- `CAPITAL_CONFIG` nu poate fi reluat de operator; se șterge numai prin `replace_capital_config`
  cu o `CapitalChangeAuthorization` validă pentru o configurație nouă (Req 13.11, 28.4).
- `DAY` nu poate fi reluat de operator; expiră la următorul `Trading_Day` (Req 13.4), consecvent
  cu `LossMonitor` (aceeași funcție `TradingDayFn`).

`KillSwitch` implementează `KillSwitchSink` (`activate`), deci primește direct activările
emise de `LossMonitor`. `state()` furnizează `KillSwitchState` pentru Risk_Engine.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterable, Iterator
from datetime import date, datetime
from enum import StrEnum
from typing import Final, Literal, Protocol, runtime_checkable

from pydantic import model_validator

from qts.config.schema import KillSwitchConfig
from qts.core.clock import Clock, ensure_utc
from qts.core.models import Frozen, UtcDatetime
from qts.persistence.journal import Journal
from qts.risk.context import KillSwitchScope, KillSwitchState
from qts.risk.monitor import (
    CapitalChangeAuthorization,
    CapitalConfigChangeError,
    KillSwitchActivation,
    TradingDayFn,
    utc_trading_day,
)

__all__ = [
    "COMPONENT",
    "COMPONENT_VERSION",
    "JOURNAL_TYPE_PREFIX",
    "ActivationOutcome",
    "CancelCommand",
    "ClearReason",
    "JournalKillSwitchStore",
    "KillSwitch",
    "KillSwitchEvent",
    "KillSwitchPersistenceError",
    "KillSwitchStore",
    "KillSwitchTrigger",
    "OpenOrder",
    "ResumeDecision",
    "ResumeRefusal",
    "ResumeRequest",
]

COMPONENT: Final = "kill_switch"
COMPONENT_VERSION: Final = "1"
JOURNAL_TYPE_PREFIX: Final = "kill_switch."

TriggerSource = Literal["manual", "automatic"]
EventKind = Literal["activated", "cleared", "resume_refused"]

# Domeniile care blochează orice instrument.
_ALL_INSTRUMENTS_SCOPES: Final = frozenset({KillSwitchScope.GLOBAL, KillSwitchScope.CAPITAL_CONFIG})


class KillSwitchPersistenceError(Exception):
    """Activarea este în vigoare în memorie, dar nu a putut fi scrisă în jurnal."""


class KillSwitchTrigger(Frozen):
    """O activare: domeniu, motiv, actor (operator sau regulă automată)."""

    activation_id: str
    scope: KillSwitchScope
    instrument: str | None = None
    reason_code: str
    detail: str = ""
    actor: str  # "operator:<nume>" sau "system:<componentă>"
    source: TriggerSource
    ts: UtcDatetime
    permanent: bool = False
    capital_config_id: str | None = None
    trading_day: date | None = None

    @model_validator(mode="after")
    def _check(self) -> KillSwitchTrigger:
        for name in ("activation_id", "reason_code", "actor"):
            if not str(getattr(self, name)).strip():
                raise ValueError(f"{name} este obligatoriu")
        if self.scope is KillSwitchScope.INSTRUMENT:
            if self.instrument is None or not self.instrument.strip():
                raise ValueError("domeniul INSTRUMENT necesită instrument")
        elif self.instrument is not None:
            raise ValueError(f"domeniul {self.scope} nu acceptă instrument")
        if self.scope is KillSwitchScope.DAY and self.trading_day is None:
            raise ValueError("domeniul DAY necesită trading_day")
        if self.scope is KillSwitchScope.CAPITAL_CONFIG and (
            not self.permanent or not (self.capital_config_id or "").strip()
        ):
            raise ValueError("domeniul CAPITAL_CONFIG este permanent și necesită capital_config_id")
        return self


class ClearReason(StrEnum):
    RESUMED = "RESUMED"
    DAY_EXPIRED = "DAY_EXPIRED"
    CAPITAL_CONFIG_REPLACED = "CAPITAL_CONFIG_REPLACED"


class ResumeRequest(Frozen):
    """Cererea de reluare pentru o activare; toate condițiile sunt verificate explicit."""

    activation_id: str
    operator: str
    approved: bool  # aprobarea explicită a operatorului (Req 14.5)
    reconciliation_ok: bool  # rezultatul reconcilierii ulterioare (Req 14.5, 14.8)
    reconciliation_id: str
    reason_resolved: bool  # motivul activării este rezolvat (Req 14.6)
    cause: str
    correction: str
    ts: UtcDatetime


class ResumeRefusal(StrEnum):
    UNKNOWN_ACTIVATION = "KS_RESUME_UNKNOWN_ACTIVATION"
    CAPITAL_CONFIG_NOT_CLEARABLE = "KS_RESUME_CAPITAL_CONFIG_NOT_CLEARABLE"
    DAY_NOT_CLEARABLE = "KS_RESUME_DAY_NOT_CLEARABLE"
    RECONCILIATION_FAILED = "KS_RESUME_RECONCILIATION_FAILED"
    APPROVAL_MISSING = "KS_RESUME_APPROVAL_MISSING"
    REASON_UNRESOLVED = "KS_RESUME_REASON_UNRESOLVED"
    CAUSE_OR_CORRECTION_MISSING = "KS_RESUME_CAUSE_OR_CORRECTION_MISSING"
    OPERATOR_MISSING = "KS_RESUME_OPERATOR_MISSING"


class ResumeDecision(Frozen):
    activation_id: str
    accepted: bool
    refusals: tuple[ResumeRefusal, ...] = ()


class KillSwitchEvent(Frozen):
    """Unitatea de persistență; starea se reconstruiește prin reluarea evenimentelor."""

    kind: EventKind
    activation_id: str
    ts: UtcDatetime
    actor: str
    trigger: KillSwitchTrigger | None = None  # pentru `activated`
    clear_reason: ClearReason | None = None  # pentru `cleared`
    resume: ResumeRequest | None = None
    refusals: tuple[ResumeRefusal, ...] = ()  # pentru `resume_refused`
    new_capital_config_id: str | None = None

    @model_validator(mode="after")
    def _check(self) -> KillSwitchEvent:
        if self.kind == "activated" and (
            self.trigger is None or self.trigger.activation_id != self.activation_id
        ):
            raise ValueError("evenimentul `activated` necesită activarea corespunzătoare")
        if self.kind == "cleared" and self.clear_reason is None:
            raise ValueError("evenimentul `cleared` necesită clear_reason")
        return self


@runtime_checkable
class KillSwitchStore(Protocol):
    def append(self, event: KillSwitchEvent) -> None: ...

    def load(self) -> Iterable[KillSwitchEvent]: ...


class JournalKillSwitchStore:
    """Persistă evenimentele `Kill_Switch` în jurnalul append-only cu lanț hash."""

    def __init__(self, journal: Journal) -> None:
        self._journal = journal

    def append(self, event: KillSwitchEvent) -> None:
        self._journal.append(
            ts=event.ts,
            type=f"{JOURNAL_TYPE_PREFIX}{event.kind}",
            correlation_id=event.activation_id,
            component=COMPONENT,
            component_version=COMPONENT_VERSION,
            actor=event.actor,
            outcome=event.kind,
            payload=event.model_dump(mode="json"),
        )

    def load(self) -> Iterator[KillSwitchEvent]:
        for record in self._journal.read():
            if record.type.startswith(JOURNAL_TYPE_PREFIX):
                yield KillSwitchEvent.model_validate(record.payload)


class OpenOrder(Protocol):
    """Ce are nevoie politica `cancel` despre un ordin deschis."""

    @property
    def client_order_id(self) -> str: ...

    @property
    def instrument(self) -> str: ...


class CancelCommand(Frozen):
    """Cerere de anulare emisă către Order_Management_Subsystem (politica `cancel`)."""

    client_order_id: str
    instrument: str
    activation_id: str
    reason_code: str


class ActivationOutcome(Frozen):
    activation_id: str
    newly_activated: bool
    policy: Literal["keep", "cancel"]
    cancel_commands: tuple[CancelCommand, ...] = ()
    kept_order_ids: tuple[str, ...] = ()


def _no_open_orders() -> Iterable[OpenOrder]:
    return ()


def _from_monitor(activation: KillSwitchActivation) -> KillSwitchTrigger:
    return KillSwitchTrigger(
        activation_id=activation.activation_id,
        scope=activation.scope,
        reason_code=str(activation.reason),
        detail=f"value={activation.value} limit={activation.limit}",
        actor="system:risk_monitor",
        source="automatic",
        ts=activation.ts,
        permanent=activation.permanent,
        capital_config_id=activation.capital_config_id,
        trading_day=activation.trading_day,
    )


class KillSwitch:
    """Starea `Kill_Switch`, thread-safe, cu verificare sincronă și persistență write-ahead."""

    def __init__(
        self,
        store: KillSwitchStore,
        *,
        clock: Clock,
        config: KillSwitchConfig | None = None,
        trading_day: TradingDayFn = utc_trading_day,
        open_orders: Callable[[], Iterable[OpenOrder]] = _no_open_orders,
    ) -> None:
        self._store = store
        self._clock = clock
        self._policy: Literal["keep", "cancel"] = (config or KillSwitchConfig()).open_orders_policy
        self._trading_day = trading_day
        self._open_orders = open_orders
        self._lock = threading.RLock()
        self._all_blocked = threading.Event()  # GLOBAL sau CAPITAL_CONFIG activ
        self._active: dict[str, KillSwitchTrigger] = {}
        self._published: tuple[KillSwitchTrigger, ...] = ()
        self._known_ids: set[str] = set()
        self._retired_capital_configs: set[str] = set()
        self._unpersisted: dict[str, KillSwitchTrigger] = {}
        self._outbox: list[CancelCommand] = []
        for event in store.load():
            self._replay(event)
        self._publish()

    # ------------------------------------------------------------------ reconstruire

    def _replay(self, event: KillSwitchEvent) -> None:
        if event.kind == "activated" and event.trigger is not None:
            self._known_ids.add(event.activation_id)
            self._active[event.activation_id] = event.trigger
        elif event.kind == "cleared":
            cleared = self._active.pop(event.activation_id, None)
            if (
                cleared is not None
                and event.clear_reason is ClearReason.CAPITAL_CONFIG_REPLACED
                and cleared.capital_config_id is not None
            ):
                self._retired_capital_configs.add(cleared.capital_config_id)

    def _publish(self) -> None:
        """Publică atomic vederea folosită de cititori (apelat sub lock)."""
        self._published = tuple(self._active.values())
        if any(t.scope in _ALL_INSTRUMENTS_SCOPES for t in self._published):
            self._all_blocked.set()
        else:
            self._all_blocked.clear()

    # ------------------------------------------------------------------ verificare sincronă

    def _day_blocks(self, trigger: KillSwitchTrigger, today: date) -> bool:
        # O zi viitoare (ceas decalat) blochează și ea: fail-closed.
        return trigger.trading_day is not None and trigger.trading_day >= today

    def blocking_scope(self, instrument: str, ts: datetime | None = None) -> KillSwitchScope | None:
        """Primul domeniu activ aplicabil instrumentului, sau None (verificare fără lock)."""
        view = self._published
        if self._all_blocked.is_set():
            for scope in (KillSwitchScope.CAPITAL_CONFIG, KillSwitchScope.GLOBAL):
                if any(t.scope is scope for t in view):
                    return scope
            return KillSwitchScope.GLOBAL  # vedere în curs de actualizare: fail-closed
        if not view:
            return None
        today = self._trading_day(ensure_utc(ts) if ts is not None else self._clock.now())
        if any(t.scope is KillSwitchScope.DAY and self._day_blocks(t, today) for t in view):
            return KillSwitchScope.DAY
        if any(t.scope is KillSwitchScope.INSTRUMENT and t.instrument == instrument for t in view):
            return KillSwitchScope.INSTRUMENT
        return None

    def allows(self, instrument: str, ts: datetime | None = None) -> bool:
        return self.blocking_scope(instrument, ts) is None

    def state(self, ts: datetime | None = None) -> KillSwitchState:
        """Instantaneul pentru `RiskContext.kill_switch`."""
        view = self._published
        today = self._trading_day(ensure_utc(ts) if ts is not None else self._clock.now())
        return KillSwitchState(
            global_active=any(t.scope is KillSwitchScope.GLOBAL for t in view),
            day_active=any(
                t.scope is KillSwitchScope.DAY and self._day_blocks(t, today) for t in view
            ),
            capital_config_active=any(t.scope is KillSwitchScope.CAPITAL_CONFIG for t in view),
            instruments=frozenset(
                t.instrument
                for t in view
                if t.scope is KillSwitchScope.INSTRUMENT and t.instrument is not None
            ),
        )

    def active_activations(self) -> tuple[KillSwitchTrigger, ...]:
        return self._published

    @property
    def open_orders_policy(self) -> Literal["keep", "cancel"]:
        return self._policy

    @property
    def unpersisted(self) -> tuple[KillSwitchTrigger, ...]:
        with self._lock:
            return tuple(self._unpersisted.values())

    # ------------------------------------------------------------------ activare

    def trigger(self, trigger: KillSwitchTrigger) -> ActivationOutcome:
        """Activează un domeniu; idempotent pe `activation_id`.

        Ridică `KillSwitchPersistenceError` dacă jurnalul nu poate fi scris; blocarea rămâne
        în vigoare, iar comenzile de anulare se emit abia după persistarea reușită.
        """
        with self._lock:
            if trigger.activation_id in self._unpersisted:
                return self._persist_activation(self._unpersisted[trigger.activation_id])
            if trigger.activation_id in self._known_ids:
                return ActivationOutcome(
                    activation_id=trigger.activation_id, newly_activated=False, policy=self._policy
                )
            if (
                trigger.scope is KillSwitchScope.CAPITAL_CONFIG
                and trigger.capital_config_id in self._retired_capital_configs
            ):
                # Reemiterea unei activări pentru o configurație retrasă nu mai are efect.
                self._known_ids.add(trigger.activation_id)
                return ActivationOutcome(
                    activation_id=trigger.activation_id, newly_activated=False, policy=self._policy
                )
            # Fail-closed: blocarea are loc înaintea oricărei scrieri.
            self._known_ids.add(trigger.activation_id)
            self._active[trigger.activation_id] = trigger
            self._publish()
            self._unpersisted[trigger.activation_id] = trigger
            return self._persist_activation(trigger)

    def _persist_activation(self, trigger: KillSwitchTrigger) -> ActivationOutcome:
        event = KillSwitchEvent(
            kind="activated",
            activation_id=trigger.activation_id,
            ts=trigger.ts,
            actor=trigger.actor,
            trigger=trigger,
        )
        try:
            self._store.append(event)
        except Exception as exc:
            raise KillSwitchPersistenceError(
                f"activarea {trigger.activation_id} nu a putut fi persistată; blocarea rămâne"
            ) from exc
        self._unpersisted.pop(trigger.activation_id, None)
        return self._apply_policy(trigger)

    def _apply_policy(self, trigger: KillSwitchTrigger) -> ActivationOutcome:
        affected = [
            o
            for o in self._open_orders()
            if trigger.scope is not KillSwitchScope.INSTRUMENT or o.instrument == trigger.instrument
        ]
        if self._policy == "keep":
            return ActivationOutcome(
                activation_id=trigger.activation_id,
                newly_activated=True,
                policy="keep",
                kept_order_ids=tuple(o.client_order_id for o in affected),
            )
        commands = tuple(
            CancelCommand(
                client_order_id=o.client_order_id,
                instrument=o.instrument,
                activation_id=trigger.activation_id,
                reason_code=trigger.reason_code,
            )
            for o in affected
        )
        self._outbox.extend(commands)
        return ActivationOutcome(
            activation_id=trigger.activation_id,
            newly_activated=True,
            policy="cancel",
            cancel_commands=commands,
        )

    def activate(self, activation: KillSwitchActivation | KillSwitchTrigger) -> None:
        """`KillSwitchSink`: primește activările automate (de exemplu de la `LossMonitor`)."""
        if isinstance(activation, KillSwitchTrigger):
            self.trigger(activation)
        else:
            self.trigger(_from_monitor(activation))

    def activate_manual(
        self,
        scope: KillSwitchScope,
        *,
        operator: str,
        reason_code: str,
        detail: str = "",
        instrument: str | None = None,
    ) -> ActivationOutcome:
        """Activarea de către operator (Req 14.1)."""
        return self._activate_new(
            scope,
            actor=f"operator:{operator}",
            source="manual",
            reason_code=reason_code,
            detail=detail,
            instrument=instrument,
        )

    def activate_automatic(
        self,
        scope: KillSwitchScope,
        *,
        component: str,
        reason_code: str,
        detail: str = "",
        instrument: str | None = None,
    ) -> ActivationOutcome:
        """Activarea printr-o regulă automată (Req 14.2): reconciliere, sănătate, audit."""
        return self._activate_new(
            scope,
            actor=f"system:{component}",
            source="automatic",
            reason_code=reason_code,
            detail=detail,
            instrument=instrument,
        )

    def _activate_new(
        self,
        scope: KillSwitchScope,
        *,
        actor: str,
        source: TriggerSource,
        reason_code: str,
        detail: str,
        instrument: str | None,
    ) -> ActivationOutcome:
        if scope is KillSwitchScope.CAPITAL_CONFIG:
            raise ValueError("CAPITAL_CONFIG se activează numai prin LossMonitor (Req 13.6)")
        now = ensure_utc(self._clock.now())
        with self._lock:
            activation_id = (
                f"{source}:{scope}:{instrument or '-'}:{now.isoformat()}:{len(self._known_ids)}"
            )
            return self.trigger(
                KillSwitchTrigger(
                    activation_id=activation_id,
                    scope=scope,
                    instrument=instrument,
                    reason_code=reason_code,
                    detail=detail,
                    actor=actor,
                    source=source,
                    ts=now,
                    trading_day=self._trading_day(now) if scope is KillSwitchScope.DAY else None,
                )
            )

    def flush(self) -> tuple[ActivationOutcome, ...]:
        """Reîncearcă persistarea activărilor nescrise; ridică la primul eșec."""
        with self._lock:
            return tuple(self._persist_activation(t) for t in list(self._unpersisted.values()))

    def drain_cancel_commands(self) -> tuple[CancelCommand, ...]:
        with self._lock:
            out = tuple(self._outbox)
            self._outbox.clear()
            return out

    # ------------------------------------------------------------------ dezactivare

    def _clear(
        self,
        activation_id: str,
        *,
        reason: ClearReason,
        actor: str,
        ts: datetime,
        resume: ResumeRequest | None = None,
        new_capital_config_id: str | None = None,
    ) -> None:
        """Persistă întâi; numai după scrierea reușită ridică blocarea (fail-closed)."""
        self._store.append(
            KillSwitchEvent(
                kind="cleared",
                activation_id=activation_id,
                ts=ts,
                actor=actor,
                clear_reason=reason,
                resume=resume,
                new_capital_config_id=new_capital_config_id,
            )
        )
        cleared = self._active.pop(activation_id)
        if reason is ClearReason.CAPITAL_CONFIG_REPLACED and cleared.capital_config_id:
            self._retired_capital_configs.add(cleared.capital_config_id)
        self._publish()

    def resume(self, request: ResumeRequest) -> ResumeDecision:
        """Reluarea unei activări (Req 14.5, 14.6, 14.8; 13.11)."""
        with self._lock:
            refusals: list[ResumeRefusal] = []
            trig = self._active.get(request.activation_id)
            if trig is None or request.activation_id in self._unpersisted:
                refusals.append(ResumeRefusal.UNKNOWN_ACTIVATION)
            elif trig.scope is KillSwitchScope.CAPITAL_CONFIG:
                refusals.append(ResumeRefusal.CAPITAL_CONFIG_NOT_CLEARABLE)
            elif trig.scope is KillSwitchScope.DAY:
                refusals.append(ResumeRefusal.DAY_NOT_CLEARABLE)
            if not request.reconciliation_ok or not request.reconciliation_id.strip():
                refusals.append(ResumeRefusal.RECONCILIATION_FAILED)
            if not request.approved:
                refusals.append(ResumeRefusal.APPROVAL_MISSING)
            if not request.reason_resolved:
                refusals.append(ResumeRefusal.REASON_UNRESOLVED)
            if not request.cause.strip() or not request.correction.strip():
                refusals.append(ResumeRefusal.CAUSE_OR_CORRECTION_MISSING)
            if not request.operator.strip():
                refusals.append(ResumeRefusal.OPERATOR_MISSING)
            actor = f"operator:{request.operator or '?'}"
            if refusals:
                self._store.append(
                    KillSwitchEvent(
                        kind="resume_refused",
                        activation_id=request.activation_id,
                        ts=request.ts,
                        actor=actor,
                        resume=request,
                        refusals=tuple(refusals),
                    )
                )
                return ResumeDecision(
                    activation_id=request.activation_id, accepted=False, refusals=tuple(refusals)
                )
            self._clear(
                request.activation_id,
                reason=ClearReason.RESUMED,
                actor=actor,
                ts=request.ts,
                resume=request,
            )
            return ResumeDecision(activation_id=request.activation_id, accepted=True)

    def expire_days(self, ts: datetime | None = None) -> tuple[str, ...]:
        """Dezactivează domeniile DAY din `Trading_Day` anterioare (Req 13.4)."""
        now = ensure_utc(ts) if ts is not None else ensure_utc(self._clock.now())
        today = self._trading_day(now)
        with self._lock:
            expired = [
                t.activation_id
                for t in self._active.values()
                if t.scope is KillSwitchScope.DAY
                and t.trading_day is not None
                and t.trading_day < today
                and t.activation_id not in self._unpersisted
            ]
            for activation_id in expired:
                self._clear(
                    activation_id, reason=ClearReason.DAY_EXPIRED, actor="system:clock", ts=now
                )
            return tuple(expired)

    def replace_capital_config(
        self, authorization: CapitalChangeAuthorization, *, actor: str
    ) -> tuple[str, ...]:
        """Singura cale de ștergere a `CAPITAL_CONFIG` (Req 13.11, 28.4)."""
        if not isinstance(authorization, CapitalChangeAuthorization):
            raise CapitalConfigChangeError("autorizarea nu respectă protocolul cerut")
        if not authorization.is_valid():
            raise CapitalConfigChangeError("Capital_Change_Authorization invalidă")
        new_id = authorization.new_capital_config_id
        now = ensure_utc(self._clock.now())
        with self._lock:
            targets = [
                t for t in self._active.values() if t.scope is KillSwitchScope.CAPITAL_CONFIG
            ]
            used = self._retired_capital_configs | {t.capital_config_id for t in targets}
            if not new_id.strip() or new_id in used:
                raise CapitalConfigChangeError(
                    f"configurația {new_id!r} nu este nouă; CAPITAL_CONFIG nu se poate "
                    "dezactiva în configurația existentă"
                )
            if any(t.activation_id in self._unpersisted for t in targets):
                raise CapitalConfigChangeError("există activări nepersistate; rulați flush()")
            for t in targets:
                self._clear(
                    t.activation_id,
                    reason=ClearReason.CAPITAL_CONFIG_REPLACED,
                    actor=actor,
                    ts=now,
                    new_capital_config_id=new_id,
                )
            return tuple(t.activation_id for t in targets)
