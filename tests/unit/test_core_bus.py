"""Teste pentru `core/bus.py`: ordonarea `(ts, prioritate_tip, seq)` și închiderea (Req 7.1)."""

from __future__ import annotations

import threading
from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from qts.core.bus import (
    BUS_PRIORITY_VERSION,
    BusClosedError,
    CommandEvent,
    EventBus,
    EventKind,
    TimerEvent,
)
from qts.core.models import Bar, ExecKind, ExecutionEvent, MarketEvent

T0 = datetime(2024, 1, 2, 9, 0, tzinfo=UTC)


def _market(ts: datetime) -> MarketEvent:
    bar = Bar(
        instrument="XYZ",
        ts_open=ts - timedelta(minutes=15),
        ts_close=ts,
        interval_min=15,
        open=Decimal(100),
        high=Decimal(100),
        low=Decimal(100),
        close=Decimal(100),
        volume=Decimal(1),
    )
    return MarketEvent(
        source_id="s", instrument="XYZ", ts_source=ts, ts_receipt=ts, kind="bar", payload=bar
    )


def _exec(ts: datetime, n: int = 1) -> ExecutionEvent:
    return ExecutionEvent(
        broker_exec_id=f"x{n}", client_order_id="c", kind=ExecKind.ACK, ts_broker=ts, ts_receipt=ts
    )


def test_priority_table_is_versioned_and_ordered() -> None:
    assert BUS_PRIORITY_VERSION == "bus-priority-v1"
    assert EventKind.EXECUTION < EventKind.MARKET < EventKind.TIMER < EventKind.COMMAND


def test_orders_by_time_then_type_priority_then_insertion() -> None:
    bus = EventBus()
    later = T0 + timedelta(minutes=1)
    bus.put(CommandEvent(command="shutdown", actor="op", ts=T0))
    bus.put(TimerEvent(name="t", ts=T0))
    bus.put(_market(T0))
    bus.put(_exec(later, 1))
    bus.put(_exec(T0, 2))
    bus.put(_exec(T0, 3))
    order = [(e.ts, e.kind) for e in bus.drain()]
    assert order == [
        (T0, EventKind.EXECUTION),
        (T0, EventKind.EXECUTION),
        (T0, EventKind.MARKET),
        (T0, EventKind.TIMER),
        (T0, EventKind.COMMAND),
        (later, EventKind.EXECUTION),
    ]


def test_equal_keys_keep_insertion_order() -> None:
    bus = EventBus()
    ids = [bus.put(_exec(T0, n)).event for n in range(5)]
    assert [e.event for e in bus.drain()] == ids


def test_peek_and_get_nowait() -> None:
    bus = EventBus()
    assert bus.peek_key() is None and bus.get_nowait() is None
    env = bus.put(_market(T0))
    assert bus.peek_key() == (T0, int(EventKind.MARKET), env.seq)
    assert bus.get_nowait() is env
    assert len(bus) == 0


def test_close_rejects_external_put_and_drains_remaining() -> None:
    bus = EventBus()
    env = bus.put(_market(T0))
    bus.close()
    with pytest.raises(BusClosedError):
        bus.put(_market(T0))
    follow_up = bus.put(_exec(T0), internal=True)  # urmări generate de motor
    assert bus.get() is follow_up and bus.get() is env
    assert bus.get() is None  # închisă și goală: nu blochează


def test_get_times_out_on_empty_bus() -> None:
    assert EventBus().get(timeout=0.01) is None


def test_get_wakes_up_on_put_from_another_thread() -> None:
    bus = EventBus()
    threading.Timer(0.05, lambda: bus.put(_market(T0))).start()
    env = bus.get(timeout=5)
    assert env is not None and env.kind is EventKind.MARKET


def test_concurrent_producers_get_unique_sequences() -> None:
    bus = EventBus()

    def produce(k: int) -> None:
        for n in range(200):
            bus.put(_exec(T0 + timedelta(seconds=n % 7), k * 1000 + n))

    threads = [threading.Thread(target=produce, args=(k,)) for k in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    out = bus.drain()
    assert len(out) == 800
    assert len({e.seq for e in out}) == 800
    assert [e.key for e in out] == sorted(e.key for e in out)
