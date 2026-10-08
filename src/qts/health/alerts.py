"""Coada persistentă de alerte, cu reîncercare și livrare în consolă/log (Req 25.4, 25.6, 26.6).

O alertă (`Alert`) este o înregistrare operațională: severitate, componentă, cod, detaliu, timp și
numărul de încercări de livrare. Spre deosebire de jurnalul append-only, tabela `alerts` din SQLite
(migrarea #3 din `persistence/db.py`) este o *coadă* actualizabilă: la fiecare încercare se
actualizează `status`, `attempts` și `last_error`. Dacă livrarea eșuează, alerta rămâne în coadă
pentru reîncercare (Req 25.6).

Canalul inițial de livrare este consola/logul (`ConsoleLogDeliverer`); un canal extern
(email/Telegram) se adaugă ulterior ca adaptor care implementează `Deliverer`. Toate detaliile trec
prin `secrets.store.Redactor` înainte de a ajunge în log, ca niciun secret să nu scape (Req 23).
"""

from __future__ import annotations

import logging
import sqlite3
from collections.abc import Callable
from datetime import datetime
from typing import Final, Protocol

from pydantic import Field

from qts.core.clock import ensure_utc
from qts.core.models import Frozen, UtcDatetime
from qts.secrets.store import REDACTOR, Redactor

__all__ = [
    "DEFAULT_MAX_ATTEMPTS",
    "Alert",
    "AlertQueue",
    "ConsoleLogDeliverer",
    "Deliverer",
    "DeliveryError",
]

DEFAULT_MAX_ATTEMPTS: Final = 5

_LOGGER = logging.getLogger("qts.health.alerts")


class DeliveryError(Exception):
    """Un `Deliverer` a eșuat la livrarea unei alerte; alerta rămâne în coadă."""


class Alert(Frozen):
    """O alertă în coadă. `id` este None până la inserare (atribuit de SQLite)."""

    id: int | None = None
    severity: str
    component: str
    code: str
    detail: str = ""
    ts: UtcDatetime
    status: str = "pending"
    attempts: int = Field(default=0, ge=0)
    last_error: str | None = None


class Deliverer(Protocol):
    """Un canal de livrare a alertelor. Ridică `DeliveryError` (sau orice excepție) la eșec."""

    def deliver(self, alert: Alert) -> None: ...


class ConsoleLogDeliverer:
    """Canalul inițial: scrie în logging/stderr, cu detaliile redactate (Req 25.6, 23)."""

    _LEVELS: Final[dict[str, int]] = {
        "info": logging.INFO,
        "warning": logging.WARNING,
        "critical": logging.CRITICAL,
    }

    def __init__(self, logger: logging.Logger | None = None, redactor: Redactor = REDACTOR) -> None:
        self._logger = logger if logger is not None else _LOGGER
        self._redactor = redactor

    def deliver(self, alert: Alert) -> None:
        level = self._LEVELS.get(alert.severity.lower(), logging.WARNING)
        detail = self._redactor.redact(alert.detail)
        self._logger.log(
            level,
            "ALERT %s/%s [%s] %s",
            self._redactor.redact(alert.component),
            self._redactor.redact(alert.code),
            self._redactor.redact(alert.severity),
            detail,
        )


class AlertQueue:
    """Coada persistentă de alerte peste tabela `alerts` (Req 25.4, 25.6).

    `enqueue` inserează o alertă `pending`. `deliver_pending` încearcă livrarea alertelor în
    așteptare prin `Deliverer`; la succes marchează `delivered`, la eșec incrementează `attempts` și
    salvează `last_error`, lăsând alerta în coadă până la `max_attempts`.
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        *,
        deliverer: Deliverer | None = None,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        redactor: Redactor = REDACTOR,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts trebuie să fie >= 1")
        self._conn = conn
        self._deliverer = (
            deliverer if deliverer is not None else ConsoleLogDeliverer(redactor=redactor)
        )
        self._max_attempts = max_attempts
        self._redactor = redactor

    @property
    def max_attempts(self) -> int:
        return self._max_attempts

    def enqueue(self, alert: Alert) -> int:
        """Inserează o alertă `pending`; întoarce id-ul atribuit."""
        detail = self._redactor.redact(alert.detail)
        cur = self._conn.execute(
            "INSERT INTO alerts (severity, component, code, detail, ts, status, attempts, "
            "last_error) VALUES (?, ?, ?, ?, ?, 'pending', 0, NULL)",
            (
                alert.severity,
                alert.component,
                alert.code,
                detail,
                ensure_utc(alert.ts).isoformat(),
            ),
        )
        return int(cur.lastrowid or 0)

    def enqueue_alert(
        self, *, severity: str, component: str, code: str, detail: str, ts: datetime
    ) -> int:
        """`AlertSink` pentru `HealthMonitor`: construiește și pune în coadă o alertă."""
        return self.enqueue(
            Alert(severity=severity, component=component, code=code, detail=detail, ts=ts)
        )

    def _row_to_alert(self, row: sqlite3.Row) -> Alert:
        return Alert(
            id=row["id"],
            severity=row["severity"],
            component=row["component"],
            code=row["code"],
            detail=row["detail"],
            ts=datetime.fromisoformat(row["ts"]),
            status=row["status"],
            attempts=row["attempts"],
            last_error=row["last_error"],
        )

    def pending(self) -> list[Alert]:
        """Alertele încă nelivrate, în ordinea inserării."""
        rows = self._conn.execute(
            "SELECT * FROM alerts WHERE status = 'pending' ORDER BY id"
        ).fetchall()
        return [self._row_to_alert(r) for r in rows]

    def get(self, alert_id: int) -> Alert | None:
        row = self._conn.execute("SELECT * FROM alerts WHERE id = ?", (alert_id,)).fetchone()
        return self._row_to_alert(row) if row is not None else None

    def mark_delivered(self, alert_id: int) -> None:
        self._conn.execute(
            "UPDATE alerts SET status = 'delivered', last_error = NULL WHERE id = ?",
            (alert_id,),
        )

    def _record_failure(self, alert_id: int, attempts: int, error: str) -> None:
        exhausted = attempts >= self._max_attempts
        self._conn.execute(
            "UPDATE alerts SET attempts = ?, last_error = ?, status = ? WHERE id = ?",
            (
                attempts,
                self._redactor.redact(error),
                "exhausted" if exhausted else "pending",
                alert_id,
            ),
        )

    def deliver_pending(self, deliverer: Deliverer | None = None) -> int:
        """Încearcă livrarea alertelor `pending`; întoarce câte au fost livrate acum.

        La succes marchează `delivered`. La eșec incrementează `attempts`, salvează `last_error` și
        lasă alerta în coadă (`pending`) până la `max_attempts`, după care devine `exhausted`.
        """
        channel = deliverer if deliverer is not None else self._deliverer
        delivered = 0
        for alert in self.pending():
            if alert.id is None:
                continue
            try:
                channel.deliver(alert)
            except Exception as exc:  # orice eșec de livrare păstrează alerta în coadă
                self._record_failure(alert.id, alert.attempts + 1, str(exc))
            else:
                self.mark_delivered(alert.id)
                delivered += 1
        return delivered


# Tip ajutător pentru teste/adaptoare care vor o funcție în loc de o clasă.
DelivererFn = Callable[[Alert], None]
