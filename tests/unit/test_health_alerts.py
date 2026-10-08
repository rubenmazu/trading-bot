"""Teste unitare pentru `health/alerts.py` (Req 25.4, 25.6, 26.6)."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest

from qts.health.alerts import (
    DEFAULT_MAX_ATTEMPTS,
    Alert,
    AlertQueue,
    ConsoleLogDeliverer,
    Deliverer,
)
from qts.persistence.db import open_db
from qts.secrets.store import Redactor

T0 = datetime(2026, 3, 2, 10, tzinfo=UTC)


def _alert(**over: object) -> Alert:
    base: dict[str, object] = {
        "severity": "warning",
        "component": "broker",
        "code": "HEALTH_BROKER_DEGRADED",
        "detail": "latență ridicată",
        "ts": T0,
    }
    base.update(over)
    return Alert.model_validate(base)


class _OkDeliverer:
    def __init__(self) -> None:
        self.delivered: list[Alert] = []

    def deliver(self, alert: Alert) -> None:
        self.delivered.append(alert)


class _FlakyDeliverer:
    """Eșuează până atinge `succeed_after` încercări, apoi reușește."""

    def __init__(self, succeed_after: int) -> None:
        self.attempts = 0
        self.succeed_after = succeed_after

    def deliver(self, alert: Alert) -> None:
        self.attempts += 1
        if self.attempts < self.succeed_after:
            raise RuntimeError(f"canal indisponibil (încercarea {self.attempts})")


# --------------------------------------------------------------------------- coadă de bază


def test_enqueue_and_pending(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "a.db")
    queue = AlertQueue(conn, redactor=Redactor())
    aid = queue.enqueue(_alert())
    assert aid > 0
    pending = queue.pending()
    assert len(pending) == 1
    assert pending[0].id == aid
    assert pending[0].status == "pending"
    assert pending[0].attempts == 0


def test_enqueue_alert_sink_interface(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "a.db")
    queue = AlertQueue(conn, redactor=Redactor())
    queue.enqueue_alert(
        severity="critical", component="storage", code="X", detail="disc plin", ts=T0
    )
    (alert,) = queue.pending()
    assert alert.component == "storage"
    assert alert.severity == "critical"


def test_mark_delivered_removes_from_pending(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "a.db")
    queue = AlertQueue(conn, redactor=Redactor())
    aid = queue.enqueue(_alert())
    queue.mark_delivered(aid)
    assert queue.pending() == []
    assert queue.get(aid) is not None
    assert queue.get(aid).status == "delivered"  # type: ignore[union-attr]


# --------------------------------------------------------------------------- livrare și reîncercare


def test_deliver_pending_delivers_all(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "a.db")
    queue = AlertQueue(conn, redactor=Redactor())
    queue.enqueue(_alert())
    queue.enqueue(_alert(code="OTHER"))
    deliverer = _OkDeliverer()
    assert queue.deliver_pending(deliverer) == 2
    assert len(deliverer.delivered) == 2
    assert queue.pending() == []


def test_failing_then_succeeding_deliverer_stays_queued_then_clears(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "a.db")
    queue = AlertQueue(conn, redactor=Redactor())
    aid = queue.enqueue(_alert())
    deliverer = _FlakyDeliverer(succeed_after=3)

    # Primele două încercări eșuează: alerta rămâne în coadă cu attempts incrementat.
    assert queue.deliver_pending(deliverer) == 0
    first = queue.get(aid)
    assert first is not None and first.status == "pending" and first.attempts == 1
    assert first.last_error is not None

    assert queue.deliver_pending(deliverer) == 0
    second = queue.get(aid)
    assert second is not None and second.attempts == 2

    # A treia reușește.
    assert queue.deliver_pending(deliverer) == 1
    done = queue.get(aid)
    assert done is not None and done.status == "delivered"
    assert queue.pending() == []


def test_exhausts_after_max_attempts(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "a.db")
    queue = AlertQueue(conn, max_attempts=2, redactor=Redactor())
    aid = queue.enqueue(_alert())

    class _AlwaysFails:
        def deliver(self, alert: Alert) -> None:
            raise RuntimeError("mereu eșuează")

    deliverer: Deliverer = _AlwaysFails()
    assert queue.deliver_pending(deliverer) == 0  # attempts -> 1, încă pending
    assert queue.get(aid).status == "pending"  # type: ignore[union-attr]
    assert queue.deliver_pending(deliverer) == 0  # attempts -> 2 == max, exhausted
    exhausted = queue.get(aid)
    assert exhausted is not None and exhausted.status == "exhausted"
    assert exhausted.attempts == 2
    assert queue.pending() == []  # nu mai e livrabilă


def test_invalid_max_attempts_rejected(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "a.db")
    with pytest.raises(ValueError):
        AlertQueue(conn, max_attempts=0)


# ----------------------------------------------------------------- redactare în consolă/log


def test_console_log_deliverer_redacts_secret(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    red = Redactor()
    red.register("super-secret-token")
    deliverer = ConsoleLogDeliverer(logger=logging.getLogger("qts.test.alerts"), redactor=red)
    alert = _alert(detail="eroare: super-secret-token")
    with caplog.at_level(logging.WARNING, logger="qts.test.alerts"):
        deliverer.deliver(alert)
    joined = " ".join(record.getMessage() for record in caplog.records)
    assert "super-secret-token" not in joined
    assert "***" in joined


def test_enqueue_redacts_secret_in_detail(tmp_path: Path) -> None:
    red = Redactor()
    red.register("super-secret-token")
    conn = open_db(tmp_path / "a.db")
    queue = AlertQueue(conn, redactor=red)
    aid = queue.enqueue(_alert(detail="token super-secret-token"))
    stored = queue.get(aid)
    assert stored is not None
    assert "super-secret-token" not in stored.detail


def test_default_max_attempts_constant(tmp_path: Path) -> None:
    conn = open_db(tmp_path / "a.db")
    queue = AlertQueue(conn, redactor=Redactor())
    assert queue.max_attempts == DEFAULT_MAX_ATTEMPTS
