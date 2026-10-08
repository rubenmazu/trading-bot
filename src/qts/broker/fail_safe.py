"""Fail_Safe_Block: ultima barieră înaintea fiecărui `BrokerAdapter.submit`.

Req 2.1, 2.5, 2.6, 15.9.

Wrapper-ul este independent de Risk_Engine și de Order_Management_Subsystem. Imediat înaintea
apelului de rețea verifică sincron, în ordine:

1. cererea are `client_order_id` și `instrument`;
2. `adapter.environment` corespunde modului aprobat în configurație (15.9), prin
   `ADAPTER_ENVIRONMENT_BY_MODE`: backtest/shadow → `sim`, demo → `demo`, live → `live`;
3. `adapter.account_id` este contul aprobat (15.9);
4. etapa proiectului (`safety/stage.py`) permite modul aprobat (2.1, 2.2); allowlist-ul pe etapă
   folosește numele modurilor din configurație;
5. pentru `live`: `Live_Gate.is_open()` (în Initial_Stage poarta este mereu închisă);
6. `Kill_Switch` nu are un domeniu activ aplicabil instrumentului (14.1, 14.2).

Orice excepție în timpul verificărilor duce la respingere (fail-closed). La respingere,
`adapter.submit` nu este apelat, se scrie un Audit_Record cu motivul (2.5) și se ridică
`FailSafeRejectedError`. Dacă scrierea auditului eșuează, cererea rămâne blocată sincron, iar
tentativa eșuată se înregistrează ca `fail_safe.rejection_audit_failed`; dacă nici aceasta nu
reușește, înregistrarea rămâne în `unaudited` pentru reîncercare prin `flush_audit()` (2.6).

Celelalte metode ale adaptorului (anulare, snapshot, evenimente) trec nemodificate prin
`__getattr__`: kill switch-ul blochează numai ordinele noi, iar anulările trebuie să ajungă la
broker pentru politica `cancel` și pentru recepția execuțiilor și reconcilierea (14.3, 14.4).

Protocolul local `GuardedBroker` descrie structural doar ce folosește bariera; adaptorul concret
din `broker/adapter.py` îl satisface, iar `broker/factory.py` (sarcina 11.5) învelește fiecare
adaptor în `FailSafeBlock`.
"""

from __future__ import annotations

import logging
from enum import StrEnum
from typing import Any, Final, NoReturn, Protocol

from qts.config.schema import AppConfig, Environment
from qts.core.clock import Clock
from qts.core.models import Frozen
from qts.risk.context import KillSwitchScope
from qts.safety.stage import ALLOWED_ENVIRONMENTS_BY_STAGE, StageInfo

__all__ = [
    "ADAPTER_ENVIRONMENT_BY_MODE",
    "COMPONENT",
    "COMPONENT_VERSION",
    "ApprovedTarget",
    "AuditAppender",
    "ClosedLiveGate",
    "FailSafeBlock",
    "FailSafeReason",
    "FailSafeRejectedError",
    "GuardedBroker",
    "LiveGateLike",
    "OrderBlocker",
    "OrderRequestLike",
]

COMPONENT: Final = "fail_safe"
COMPONENT_VERSION: Final = "1"
_LOG = logging.getLogger(__name__)


class FailSafeReason(StrEnum):
    INVALID_REQUEST = "FS_INVALID_REQUEST"
    ENVIRONMENT_MISMATCH = "FS_ENVIRONMENT_MISMATCH"
    ACCOUNT_MISMATCH = "FS_ACCOUNT_MISMATCH"
    STAGE_FORBIDS_ENVIRONMENT = "FS_STAGE_FORBIDS_ENVIRONMENT"
    LIVE_GATE_CLOSED = "FS_LIVE_GATE_CLOSED"
    KILL_SWITCH_ACTIVE = "FS_KILL_SWITCH_ACTIVE"
    CHECK_FAILED = "FS_CHECK_FAILED"


class FailSafeRejectedError(Exception):
    """Cererea a fost oprită înaintea adaptorului."""

    def __init__(
        self, reason: FailSafeReason, detail: str, *, client_order_id: str, audited: bool
    ) -> None:
        self.reason = reason
        self.detail = detail
        self.client_order_id = client_order_id
        self.audited = audited
        super().__init__(f"{reason}: {detail}")


class OrderRequestLike(Protocol):
    @property
    def client_order_id(self) -> str: ...

    @property
    def instrument(self) -> str: ...


class GuardedBroker[ReqT: OrderRequestLike, AckT](Protocol):
    """Ce folosește bariera din adaptor; restul interfeței trece nemodificat."""

    @property
    def environment(self) -> str: ...

    @property
    def account_id(self) -> str | None: ...

    def submit(self, req: ReqT, /) -> AckT: ...


class LiveGateLike(Protocol):
    def is_open(self) -> bool: ...


class ClosedLiveGate:
    """Poarta implicită: închisă (Initial_Stage)."""

    def is_open(self) -> bool:
        return False


class OrderBlocker(Protocol):
    def blocking_scope(self, instrument: str) -> KillSwitchScope | None: ...


class AuditAppender(Protocol):
    """Subsetul din `Journal.append` folosit pentru audit."""

    def append(
        self,
        *,
        ts: Any,
        type: str,
        correlation_id: str,
        component: str,
        component_version: str,
        actor: str,
        outcome: str,
        payload: dict[str, Any],
    ) -> object: ...


# Mediul raportat de adaptor (`broker/adapter.py`: sim|demo|live) pentru fiecare mod din
# configurație (backtest|shadow|demo|live). Backtest și Shadow rulează pe adaptorul simulat.
ADAPTER_ENVIRONMENT_BY_MODE: Final[dict[str, str]] = {
    "backtest": "sim",
    "shadow": "sim",
    "demo": "demo",
    "live": "live",
}


class ApprovedTarget(Frozen):
    """Mediul (modul din configurație) și contul aprobate în configurație."""

    environment: Environment
    account_id: str | None = None

    @classmethod
    def from_config(cls, config: AppConfig) -> ApprovedTarget:
        return cls(environment=config.environment, account_id=config.broker.account_id)

    @property
    def adapter_environment(self) -> str:
        """Mediul pe care trebuie să îl raporteze adaptorul pentru modul aprobat."""
        return ADAPTER_ENVIRONMENT_BY_MODE[self.environment]


class FailSafeBlock[ReqT: OrderRequestLike, AckT]:
    """Învelește un adaptor; numai `submit` este păzit, restul este delegat."""

    def __init__(
        self,
        inner: GuardedBroker[ReqT, AckT],
        *,
        approved: ApprovedTarget,
        stage: StageInfo,
        kill_switch: OrderBlocker,
        audit: AuditAppender,
        clock: Clock,
        live_gate: LiveGateLike | None = None,
    ) -> None:
        self._inner = inner
        self._approved = approved
        self._stage = stage
        self._kill_switch = kill_switch
        self._audit = audit
        self._clock = clock
        self._live_gate: LiveGateLike = live_gate or ClosedLiveGate()
        self._unaudited: list[dict[str, Any]] = []

    # ------------------------------------------------------------------ interfață

    @property
    def inner(self) -> GuardedBroker[ReqT, AckT]:
        return self._inner

    @property
    def environment(self) -> str:
        return self._inner.environment

    @property
    def account_id(self) -> str | None:
        return self._inner.account_id

    @property
    def unaudited(self) -> tuple[dict[str, Any], ...]:
        return tuple(self._unaudited)

    def __getattr__(self, name: str) -> Any:
        # Apelat numai pentru atributele care lipsesc din wrapper: cancel, snapshot, events...
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def submit(self, req: ReqT, /) -> AckT:
        client_order_id = "?"
        try:
            client_order_id = str(req.client_order_id)
            verdict = self._check(req)
        except Exception as exc:
            verdict = (FailSafeReason.CHECK_FAILED, f"verificare eșuată: {type(exc).__name__}")
        if verdict is not None:
            reason, detail = verdict
            try:
                self._reject(reason, detail, client_order_id, req)
            except FailSafeRejectedError:
                raise
            except Exception as exc:
                # Calea de respingere a eșuat: cererea rămâne blocată sincron (Req 2.6).
                self._unaudited.append(
                    {
                        "ts": None,
                        "payload": {
                            "reason_code": str(reason),
                            "detail": detail,
                            "client_order_id": client_order_id,
                        },
                    }
                )
                raise FailSafeRejectedError(
                    reason, detail, client_order_id=client_order_id, audited=False
                ) from exc
        return self._inner.submit(req)

    # ------------------------------------------------------------------ verificări

    def _check(self, req: ReqT) -> tuple[FailSafeReason, str] | None:
        if not str(req.client_order_id).strip() or not str(req.instrument).strip():
            return FailSafeReason.INVALID_REQUEST, "client_order_id și instrument sunt obligatorii"
        env = self._inner.environment
        mode = self._approved.environment
        if env != self._approved.adapter_environment:
            return (
                FailSafeReason.ENVIRONMENT_MISMATCH,
                f"adaptorul este în {env!r}, configurația aprobată: {mode!r}",
            )
        if self._inner.account_id != self._approved.account_id:
            # Valorile conturilor nu sunt incluse în motiv.
            return FailSafeReason.ACCOUNT_MISMATCH, "contul adaptorului diferă de cel aprobat"
        # Etapa se verifică pe modul din configurație (aceleași nume ca în `safety/stage.py`).
        allowed = ALLOWED_ENVIRONMENTS_BY_STAGE.get(self._stage.stage, frozenset())
        if mode not in allowed:
            return (
                FailSafeReason.STAGE_FORBIDS_ENVIRONMENT,
                f"modul {mode!r} nu este permis în etapa {self._stage.stage}",
            )
        if env == "live" and self._live_gate.is_open() is not True:
            return FailSafeReason.LIVE_GATE_CLOSED, "Live_Gate este închis"
        scope = self._kill_switch.blocking_scope(str(req.instrument))
        if scope is not None:
            return FailSafeReason.KILL_SWITCH_ACTIVE, f"Kill_Switch activ: {scope}"
        return None

    def _reject(
        self, reason: FailSafeReason, detail: str, client_order_id: str, req: ReqT
    ) -> NoReturn:
        try:
            instrument = str(req.instrument)
        except Exception:
            instrument = "?"
        try:
            ts = self._clock.now()
        except Exception:
            ts = None
        payload: dict[str, Any] = {
            "reason_code": str(reason),
            "detail": detail,
            "client_order_id": client_order_id,
            "instrument": instrument,
            "approved_environment": self._approved.environment,
            "stage": str(self._stage.stage),
        }
        audited = self._write(ts, "fail_safe.rejected", client_order_id, "rejected", payload)
        if not audited:
            failed = {**payload, "audit_error": "scrierea Audit_Record a eșuat"}
            if not self._write(
                ts, "fail_safe.rejection_audit_failed", client_order_id, "blocked", failed
            ):
                self._unaudited.append({"ts": ts, "payload": failed})
                _LOG.critical(
                    "Fail_Safe_Block: respingere neauditată (%s, %s)", reason, client_order_id
                )
        raise FailSafeRejectedError(
            reason, detail, client_order_id=client_order_id, audited=audited
        )

    def _write(
        self,
        ts: Any,
        type_: str,
        correlation_id: str,
        outcome: str,
        payload: dict[str, Any],
    ) -> bool:
        if ts is None:
            return False
        try:
            self._audit.append(
                ts=ts,
                type=type_,
                correlation_id=correlation_id if correlation_id.strip() else "?",
                component=COMPONENT,
                component_version=COMPONENT_VERSION,
                actor="system:fail_safe",
                outcome=outcome,
                payload=payload,
            )
        except Exception:
            return False
        return True

    def flush_audit(self) -> int:
        """Reîncearcă scrierea respingerilor neauditate; întoarce câte au rămas."""
        remaining: list[dict[str, Any]] = []
        for item in self._unaudited:
            ts = item["ts"]
            if ts is None:
                try:
                    ts = self._clock.now()
                except Exception:
                    ts = None
            payload = item["payload"]
            if not self._write(
                ts,
                "fail_safe.rejection_audit_failed",
                str(payload.get("client_order_id", "?")),
                "blocked",
                payload,
            ):
                remaining.append(item)
        self._unaudited = remaining
        return len(remaining)
