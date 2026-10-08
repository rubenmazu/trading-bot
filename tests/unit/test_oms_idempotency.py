"""Teste unitare pentru `oms/idempotency.py` (Req 10.1, 10.2, 10.5)."""

from datetime import UTC, datetime
from decimal import Decimal

import pytest

from qts.core.models import ExecKind, ExecutionEvent
from qts.oms.idempotency import (
    KEY_LENGTH,
    AppliedExec,
    IdempotencyRegistry,
    exec_fingerprint,
    idempotency_key,
)

T0 = datetime(2026, 1, 5, 9, tzinfo=UTC)


def test_key_is_deterministic_and_fixed_length() -> None:
    a = idempotency_key("run1", "mr", "XYZ", 7)
    assert a == idempotency_key("run1", "mr", "XYZ", 7)
    assert len(a) == KEY_LENGTH
    assert all(c in "0123456789abcdef" for c in a)


@pytest.mark.parametrize(
    "args",
    [
        ("run2", "mr", "XYZ", 7),
        ("run1", "mr2", "XYZ", 7),
        ("run1", "mr", "ABC", 7),
        ("run1", "mr", "XYZ", 8),
    ],
)
def test_key_changes_with_each_component(args: tuple[str, str, str, int]) -> None:
    assert idempotency_key(*args) != idempotency_key("run1", "mr", "XYZ", 7)


def test_key_components_are_not_ambiguous() -> None:
    # concatenarea naivă ar confunda ("ab", "c") cu ("a", "bc")
    assert idempotency_key("ab", "c", "X", 1) != idempotency_key("a", "bc", "X", 1)


@pytest.mark.parametrize(
    "args", [("", "mr", "XYZ", 1), ("r", " ", "XYZ", 1), ("r", "mr", "XYZ", -1)]
)
def test_key_rejects_invalid_input(args: tuple[str, str, str, int]) -> None:
    with pytest.raises(ValueError):
        idempotency_key(*args)


def _event(**over: object) -> ExecutionEvent:
    data: dict[str, object] = {
        "broker_exec_id": "e1",
        "client_order_id": "c1",
        "kind": ExecKind.PARTIAL_FILL,
        "qty": Decimal(1),
        "price": Decimal(10),
        "ts_broker": T0,
        "ts_receipt": T0,
    }
    data.update(over)
    return ExecutionEvent.model_validate(data)


def test_fingerprint_ignores_receipt_time_only() -> None:
    base = exec_fingerprint(_event())
    assert exec_fingerprint(_event(ts_receipt=datetime(2026, 1, 6, tzinfo=UTC))) == base
    assert exec_fingerprint(_event(qty=Decimal(2))) != base


def test_registry_orders_and_execs() -> None:
    reg = IdempotencyRegistry()
    assert reg.register_order("c1") is True
    assert reg.register_order("c1") is False
    assert reg.has_order("c1")
    entry = AppliedExec("e1", "c1", 3, "fp")
    reg.mark_applied(entry)
    assert reg.applied("e1") == entry
    with pytest.raises(ValueError):
        reg.mark_applied(entry)
    restored = IdempotencyRegistry(order_keys=["c1"], applied=[entry])
    assert restored.has_order("c1") and restored.applied("e1") == entry
