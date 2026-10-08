"""Health_Monitor: stări per componentă, ceas vs. NTP, blocarea ordinelor noi (Req 25, 26.6, 26.7).

Agregă `Health_State` pentru componentele enumerate în `Component` (date, broker, reconciliere,
Risk_Engine, stocare, ceas, audit). Fiecare componentă are o stare (`ComponentState`), un motiv și
timpul ultimei schimbări; starea globală este cea mai severă stare individuală (Req 25.1, 25.2).

Comportament fail-closed, consecvent cu `safety/kill_switch.py` și `safety/stage.py`:
- o componentă `CRITICAL` în Demo sau Live activează `Kill_Switch(GLOBAL)` și blochează ordinele
  noi; blocarea este sincronă (nu depinde de coada de evenimente), deci efectul este sub 1 s
  (Req 25.3). În Backtest/Shadow nu se activează kill switch-ul, dar `blocks_new_orders()` reflectă
  oricum starea pentru porțile de pornire (Req 26.7);
- ceasul se compară periodic cu NTP (`check_clock`): o deviație peste 500 ms în Demo/Live suspendă
  ordinele noi (Req 26.6) marcând componenta `CLOCK` ca `DEGRADED`; sursa NTP absentă/indisponibilă
  lasă `CLOCK` în `UNKNOWN` care, în Demo/Live, suspendă și ea (fail-closed);
- sursa NTP este injectată (un `Callable[[], datetime] | None`); modulul nu face apeluri de rețea.

Pragul de 500 ms este o constantă versionată (`CLOCK_DEVIATION_LIMIT_MS`, `CLOCK_LIMIT_VERSION`).

`HealthMonitor` emite o alertă la fiecare schimbare de stare (Req 25.4) printr-un sink injectat
(`AlertSink`, satisfăcut de `health/alerts.AlertQueue`); detaliile trec prin `Redactor` înainte de
a ajunge la coadă/log (Req 23).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from enum import IntEnum, StrEnum
from typing import Final, Protocol

from qts.core.clock import Clock, ensure_utc
from qts.risk.context import KillSwitchScope
from qts.safety.kill_switch import KillSwitch
from qts.secrets.store import REDACTOR, Redactor

__all__ = [
    "CLOCK_DEVIATION_LIMIT_MS",
    "CLOCK_LIMIT_VERSION",
    "COMPONENT",
    "COMPONENT_VERSION",
    "SUSPENDING_MODES",
    "AlertSink",
    "Component",
    "ComponentState",
    "ComponentStatus",
    "HealthMonitor",
    "HealthReport",
]

COMPONENT: Final = "health"
COMPONENT_VERSION: Final = "1"

# Pragul deviației ceasului (Req 26.6). Versionat: îl schimbăm doar cu o migrare de versiune.
CLOCK_LIMIT_VERSION: Final = "2026-10-06.1"
CLOCK_DEVIATION_LIMIT_MS: Final = 500

# Modurile în care comportamentul fail-closed suspendă ordinele noi (Req 26.6, 25.3).
SUSPENDING_MODES: Final[frozenset[str]] = frozenset({"demo", "live"})


class ComponentState(IntEnum):
    """Starea unei componente, cu ordonare stabilă după severitate (crescătoare).

    `IntEnum` garantează ordonarea: `HEALTHY < DEGRADED < UNKNOWN < CRITICAL`. Starea globală este
    `max()` peste stările componentelor. `UNKNOWN` este mai sever decât `DEGRADED` (fail-closed:
    necunoscutul se tratează ca potențial periculos), dar sub `CRITICAL`.
    """

    HEALTHY = 0
    DEGRADED = 1
    UNKNOWN = 2
    CRITICAL = 3

    @property
    def label(self) -> str:
        return self.name


class Component(StrEnum):
    DATA = "data"
    BROKER = "broker"
    RECONCILIATION = "reconciliation"
    RISK = "risk"
    STORAGE = "storage"
    CLOCK = "clock"
    AUDIT = "audit"


class ComponentStatus(Protocol):
    @property
    def state(self) -> ComponentState: ...

    @property
    def reason(self) -> str: ...

    @property
    def ts(self) -> datetime: ...


class _Status:
    __slots__ = ("_reason", "_state", "_ts")

    def __init__(self, state: ComponentState, reason: str, ts: datetime) -> None:
        self._state = state
        self._reason = reason
        self._ts = ts

    @property
    def state(self) -> ComponentState:
        return self._state

    @property
    def reason(self) -> str:
        return self._reason

    @property
    def ts(self) -> datetime:
        return self._ts


class HealthReport:
    """Instantaneu imuabil al `Health_State` (Req 25.1)."""

    __slots__ = ("_mode", "_overall", "_statuses")

    def __init__(
        self, statuses: dict[Component, _Status], overall: ComponentState, mode: str
    ) -> None:
        self._statuses = statuses
        self._overall = overall
        self._mode = mode

    @property
    def overall(self) -> ComponentState:
        return self._overall

    @property
    def mode(self) -> str:
        return self._mode

    def status(self, component: Component) -> ComponentStatus:
        return self._statuses[component]

    @property
    def components(self) -> dict[Component, ComponentStatus]:
        return dict(self._statuses)

    def healthy(self) -> bool:
        return self._overall is ComponentState.HEALTHY

    def blocks_new_orders(self) -> bool:
        """True dacă starea curentă trebuie să suspende ordinele noi (Req 25.3, 26.6, 26.7)."""
        if self._overall is ComponentState.CRITICAL:
            return True
        if self._mode in SUSPENDING_MODES:
            return self._overall >= ComponentState.DEGRADED
        return False


class AlertSink(Protocol):
    """Minimul cerut de `HealthMonitor` de la o coadă de alerte (vezi `health/alerts`)."""

    def enqueue_alert(
        self, *, severity: str, component: str, code: str, detail: str, ts: datetime
    ) -> int: ...


class HealthMonitor:
    """Agregă stările componentelor și aplică politica fail-closed (Req 25, 26.6, 26.7).

    `mode` este `Operational_Mode` curent ("backtest", "shadow", "demo", "live"); numai în Demo/Live
    deviația ceasului / `UNKNOWN` suspendă și kill switch-ul se activează la `CRITICAL`.
    """

    def __init__(
        self,
        *,
        clock: Clock,
        mode: str,
        kill_switch: KillSwitch,
        alerts: AlertSink | None = None,
        redactor: Redactor = REDACTOR,
    ) -> None:
        self._clock = clock
        self._mode = mode
        self._kill_switch = kill_switch
        self._alerts = alerts
        self._redactor = redactor
        now = ensure_utc(clock.now())
        # Toate componentele pornesc UNKNOWN: necunoscutul e fail-closed până la prima observație.
        self._statuses: dict[Component, _Status] = {
            component: _Status(ComponentState.UNKNOWN, "neobservat", now) for component in Component
        }

    # ------------------------------------------------------------------ stare per componentă

    def set_state(
        self,
        component: Component,
        state: ComponentState,
        *,
        reason: str,
        ts: datetime | None = None,
    ) -> None:
        """Setează starea unei componente; emite alertă și aplică kill switch la schimbare.

        La o schimbare de stare înregistrează componenta, starea nouă, cauza și timpul (Req 25.2)
        și emite o alertă (Req 25.4). O componentă `CRITICAL` în Demo/Live activează
        `Kill_Switch(GLOBAL)` (Req 25.3).
        """
        at = ensure_utc(ts) if ts is not None else ensure_utc(self._clock.now())
        previous = self._statuses[component]
        if previous.state is state and previous.reason == reason:
            return
        self._statuses[component] = _Status(state, reason, at)
        self._emit_alert(component, state, reason, at)
        if state is ComponentState.CRITICAL and self._mode in SUSPENDING_MODES:
            self._kill_switch.activate_automatic(
                KillSwitchScope.GLOBAL,
                component=COMPONENT,
                reason_code=f"HEALTH_CRITICAL_{component.value.upper()}",
                detail=self._redactor.redact(f"{component.value}: {reason}"),
            )

    def state_of(self, component: Component) -> ComponentState:
        return self._statuses[component].state

    # ------------------------------------------------------------------ ceas vs. NTP

    def check_clock(
        self,
        *,
        ntp_now: Callable[[], datetime] | None = None,
        local_now: datetime | None = None,
    ) -> ComponentState:
        """Compară ceasul local cu NTP și actualizează componenta `CLOCK` (Req 26.6).

        `ntp_now` este o sursă injectată (fără apeluri de rețea). `local_now` implicit vine din
        `core.clock.Clock`. Absența/indisponibilitatea NTP lasă `CLOCK` în `UNKNOWN`; o deviație
        peste `CLOCK_DEVIATION_LIMIT_MS` o marchează `DEGRADED`; sub prag o face `HEALTHY`.
        """
        local = ensure_utc(local_now) if local_now is not None else ensure_utc(self._clock.now())
        if ntp_now is None:
            self.set_state(
                Component.CLOCK,
                ComponentState.UNKNOWN,
                reason="sursă NTP indisponibilă",
                ts=local,
            )
            return ComponentState.UNKNOWN
        try:
            raw_ntp = ntp_now()
        except Exception as exc:  # orice eșec al sursei NTP este fail-closed (UNKNOWN)
            self.set_state(
                Component.CLOCK,
                ComponentState.UNKNOWN,
                reason=f"NTP indisponibil: {type(exc).__name__}",
                ts=local,
            )
            return ComponentState.UNKNOWN
        # Un timp NTP fără fus orar este o eroare de contract, nu o indisponibilitate: se propagă.
        ntp = ensure_utc(raw_ntp)
        deviation_ms = abs((local - ntp).total_seconds()) * 1000.0
        if deviation_ms > CLOCK_DEVIATION_LIMIT_MS:
            self.set_state(
                Component.CLOCK,
                ComponentState.DEGRADED,
                reason=(
                    f"deviație ceas {deviation_ms:.0f} ms > {CLOCK_DEVIATION_LIMIT_MS} ms "
                    f"(prag {CLOCK_LIMIT_VERSION})"
                ),
                ts=local,
            )
            return ComponentState.DEGRADED
        self.set_state(
            Component.CLOCK,
            ComponentState.HEALTHY,
            reason=f"deviație ceas {deviation_ms:.0f} ms în limită",
            ts=local,
        )
        return ComponentState.HEALTHY

    # ------------------------------------------------------------------ agregare și porți

    def report(self) -> HealthReport:
        overall = max((s.state for s in self._statuses.values()), default=ComponentState.HEALTHY)
        return HealthReport(dict(self._statuses), overall, self._mode)

    def overall(self) -> ComponentState:
        return self.report().overall

    def healthy(self) -> bool:
        return self.report().healthy()

    def blocks_new_orders(self) -> bool:
        """Blochează/suspendă ordinele noi conform `HealthReport` (Req 25.3, 26.6, 26.7)."""
        return self.report().blocks_new_orders()

    def startup_blocked(self) -> bool:
        """Poarta de pornire: o componentă critică blochează pornirea/ordinele noi (Req 26.7)."""
        return self.blocks_new_orders()

    # ------------------------------------------------------------------ alerte

    def _severity_of(self, state: ComponentState) -> str:
        return {
            ComponentState.HEALTHY: "info",
            ComponentState.DEGRADED: "warning",
            ComponentState.UNKNOWN: "warning",
            ComponentState.CRITICAL: "critical",
        }[state]

    def _emit_alert(
        self, component: Component, state: ComponentState, reason: str, ts: datetime
    ) -> None:
        if self._alerts is None:
            return
        self._alerts.enqueue_alert(
            severity=self._severity_of(state),
            component=component.value,
            code=f"HEALTH_{component.value.upper()}_{state.label}",
            detail=self._redactor.redact(reason),
            ts=ts,
        )
