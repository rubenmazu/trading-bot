"""Infrastructura suitei contract pentru `BrokerAdapter` (Req 3.1–3.4).

Suita din `test_broker_contract.py` rulează aceleași verificări pe fiecare adaptor înregistrat
în `ADAPTER_FACTORIES`. Un adaptor este descris de o fabrică ce întoarce un `Harness` nou:

- `adapter`: instanța care satisface `BrokerAdapter`, cu instrumentul `CONTRACT_SYMBOL`
  (EUR, tick 0.01, pas 1, fără fracționare) disponibil și fără poziții inițiale;
- `step()`: „driverul” pieței. Avansează piața cu o unitate (o bară, un tick, un mesaj) și face
  adaptorul să execute fiecare ordin deschis cu cel mult `FILL_CHUNK` unități, la un preț > 0.
  Pentru `CONTRACT_QTY = 10` rezultă execuțiile 4, 4, 2 (PARTIAL_FILL, PARTIAL_FILL, FILL);
- `clock`: ceasul injectat, avansat de driver.

Înregistrarea unui adaptor nou (de exemplu adaptorul demo concret)
    1. Scrieți `def _demo_harness(caps: BrokerCapabilities | None) -> Harness`. Dacă `caps`
       nu este `None`, adaptorul trebuie să declare exact aceste capabilități; dacă brokerul
       nu permite asta, apelați `pytest.skip(...)` pentru testele care cer restricții.
    2. Implementați `step()` pe mediul demo (de exemplu, așteptarea execuțiilor brokerului
       sau un endpoint de test care le forțează) respectând contractul de mai sus.
    3. Adăugați intrarea `"demo": _demo_harness` în `ADAPTER_FACTORIES`. Testele se
       parametrizează automat după cheile dicționarului; nu se modifică testele.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import Final

from qts.broker.adapter import BrokerAdapter, BrokerCapabilities, BrokerSnapshot
from qts.broker.fake import DEFAULT_FAKE_CAPABILITIES, FakeBroker
from qts.broker.sim import DEFAULT_SIM_CAPABILITIES, SimBarContext, SimBroker, SimBrokerConfig
from qts.core.clock import SimClock
from qts.core.models import Bar, ExecKind, ExecutionEvent, Instrument, OrderState
from qts.core.money import ZERO
from qts.costs.commission_tables import CommissionSchedule, CommissionTable
from qts.costs.model import (
    CompleteCostModel,
    CostModelConfig,
    FxConfig,
    LatencyConfig,
    SlippageConfig,
    SpreadSchedule,
    TaxConfig,
)

__all__ = [
    "ADAPTER_FACTORIES",
    "CONTRACT_QTY",
    "CONTRACT_SYMBOL",
    "FILL_CHUNK",
    "T0",
    "AdapterFactory",
    "Harness",
    "OrderSummary",
    "check_event_stream",
    "check_snapshot",
    "contract_instrument",
]

D = Decimal
T0: Final = datetime(2025, 1, 2, 9, 0, tzinfo=UTC)
STEP: Final = timedelta(minutes=15)
CONTRACT_SYMBOL: Final = "XYZ"
CONTRACT_QTY: Final = D(10)
FILL_CHUNK: Final = D(4)
_PRICE: Final = D(10)

_FILL_KINDS: Final = frozenset({ExecKind.PARTIAL_FILL, ExecKind.FILL})
_TERMINAL_KINDS: Final = frozenset(
    {ExecKind.FILL, ExecKind.CANCELLED, ExecKind.EXPIRED, ExecKind.REJECT}
)
_STATE_OF: Final = {
    ExecKind.ACK: OrderState.ACKNOWLEDGED,
    ExecKind.REJECT: OrderState.REJECTED_BROKER,
    ExecKind.PARTIAL_FILL: OrderState.PARTIALLY_FILLED,
    ExecKind.FILL: OrderState.FILLED,
    ExecKind.CANCELLED: OrderState.CANCELLED,
    ExecKind.EXPIRED: OrderState.EXPIRED,
}
_OPEN_STATES: Final = frozenset({OrderState.ACKNOWLEDGED, OrderState.PARTIALLY_FILLED})


def contract_instrument() -> Instrument:
    return Instrument(
        symbol=CONTRACT_SYMBOL,
        venue="XETR",
        asset_class="etf",
        currency="EUR",
        tick_size=D("0.01"),
        qty_step=D(1),
        min_qty=D(1),
        calendar_id="XETR",
    )


# --------------------------------------------------------------------------- harness


@dataclass(slots=True)
class Harness:
    """Un adaptor proaspăt, driverul pieței și jurnalul evenimentelor livrate."""

    name: str
    adapter: BrokerAdapter
    step: Callable[[], None]
    clock: SimClock
    log: list[ExecutionEvent] = field(default_factory=list)

    def drain(self) -> list[ExecutionEvent]:
        """Consumă evenimentele livrate de adaptor și le adaugă în jurnal."""
        batch = list(self.adapter.events())
        self.log.extend(batch)
        return batch

    def step_until_terminal(self, client_order_id: str, max_steps: int = 10) -> None:
        for _ in range(max_steps):
            states = {o.client_order_id: o.state for o in self.adapter.snapshot().orders}
            if states.get(client_order_id) not in _OPEN_STATES:
                return
            self.step()
        raise AssertionError(f"{self.name}: ordinul {client_order_id} nu s-a încheiat")


AdapterFactory = Callable[[BrokerCapabilities | None], Harness]


# --------------------------------------------------------------------------- SimBroker


def _sim_cost_model() -> CompleteCostModel:
    table = CommissionTable(
        broker="sim",
        version="v1",
        valid_from=datetime(2024, 1, 1, tzinfo=UTC),
        currency="EUR",
        percent=D("0.001"),
        minimum=D(1),
    )
    return CompleteCostModel(
        CostModelConfig(
            version="costs-v1",
            commissions=CommissionSchedule(tables=(table,)),
            spreads={CONTRACT_SYMBOL: SpreadSchedule(default=D("0.002"))},
            slippage=SlippageConfig(k=D(0), min_ticks=D(1)),
            latency=LatencyConfig(latency_ms=500),
            fx=FxConfig(conversion_spread=D("0.002")),
            taxes=TaxConfig(approved=True, reference="operator"),
        )
    )


def _sim_harness(caps: BrokerCapabilities | None) -> Harness:
    clock = SimClock(T0)
    broker = SimBroker(
        account_id="SIM-CONTRACT",
        instruments=[contract_instrument()],
        cost_model=_sim_cost_model(),
        clock=clock,
        config=SimBrokerConfig(
            max_fill_qty_per_bar=FILL_CHUNK,
            capabilities=caps if caps is not None else DEFAULT_SIM_CAPABILITIES,
        ),
    )
    ctx = SimBarContext(sigma_bar=D("0.01"), adv=D(10000))

    def step() -> None:
        # Bara începe la momentul curent, deci ordinele trimise până acum sunt eligibile.
        start = clock.now()
        bar = Bar(
            instrument=CONTRACT_SYMBOL,
            ts_open=start,
            ts_close=start + STEP,
            interval_min=15,
            open=_PRICE,
            high=_PRICE + 1,
            low=_PRICE - 1,
            close=_PRICE,
            volume=D(1000),
        )
        broker.on_bar(bar, ctx)
        clock.advance_to(start + STEP)

    return Harness("sim", broker, step, clock)


# --------------------------------------------------------------------------- FakeBroker


def _fake_harness(caps: BrokerCapabilities | None) -> Harness:
    clock = SimClock(T0)
    broker = FakeBroker(
        instruments=[contract_instrument()],
        clock=clock,
        account_id="FAKE-CONTRACT",
        capabilities=caps if caps is not None else DEFAULT_FAKE_CAPABILITIES,
    )

    def step() -> None:
        for status in broker.truth().orders:
            if status.state in _OPEN_STATES:
                qty = min(FILL_CHUNK, status.remaining_qty)
                broker.fill(status.client_order_id, qty, _PRICE)
        clock.advance_to(clock.now() + STEP)

    return Harness("fake", broker, step, clock)


ADAPTER_FACTORIES: Final[dict[str, AdapterFactory]] = {
    "sim": _sim_harness,
    "fake": _fake_harness,
}


# --------------------------------------------------------------------------- verificări


@dataclass(frozen=True, slots=True)
class OrderSummary:
    events: tuple[ExecutionEvent, ...]
    filled_qty: Decimal
    last_state: OrderState


def check_event_stream(
    events: Iterable[ExecutionEvent], order_qty: dict[str, Decimal]
) -> dict[str, OrderSummary]:
    """Verifică invariantele de flux (Req 3.1) și întoarce rezumatul pe ordin.

    Fără defecte injectate, fiecare ordin are `seq` contiguu de la 1, primul mesaj este ACK
    (acceptat) sau REJECT (respins), execuțiile au qty/price > 0 și nu depășesc cantitatea,
    iar după un mesaj terminal mai poate sosi numai `CANCEL_REJECTED`.
    """
    by_order: dict[str, list[ExecutionEvent]] = defaultdict(list)
    seen_ids: set[str] = set()
    for ev in events:
        assert isinstance(ev, ExecutionEvent)
        assert ev.broker_exec_id, "broker_exec_id gol"
        assert ev.broker_exec_id not in seen_ids, f"broker_exec_id duplicat {ev.broker_exec_id}"
        seen_ids.add(ev.broker_exec_id)
        by_order[ev.client_order_id].append(ev)

    out: dict[str, OrderSummary] = {}
    for coid, evs in by_order.items():
        assert coid in order_qty, f"eveniment pentru ordin netrimis {coid}"
        assert [e.seq for e in evs] == list(range(1, len(evs) + 1)), f"{coid}: seq necontiguu"
        assert evs[0].kind in (ExecKind.ACK, ExecKind.REJECT), f"{coid}: primul mesaj {evs[0]}"
        filled = ZERO
        state = _STATE_OF[evs[0].kind]
        terminal = evs[0].kind is ExecKind.REJECT
        for e in evs[1:]:
            if terminal:
                assert e.kind is ExecKind.CANCEL_REJECTED, f"{coid}: {e.kind} după terminal"
                continue
            assert e.kind not in (ExecKind.ACK, ExecKind.REJECT), f"{coid}: {e.kind} repetat"
            if e.kind in _FILL_KINDS:
                assert e.qty is not None and e.qty > 0
                assert e.price is not None and e.price > 0
                filled += e.qty
                assert filled <= order_qty[coid], f"{coid}: supra-execuție"
                expected = ExecKind.FILL if filled == order_qty[coid] else ExecKind.PARTIAL_FILL
                assert e.kind is expected, f"{coid}: {e.kind} la cumulat {filled}"
            elif e.kind is ExecKind.CANCELLED:
                assert e.qty == order_qty[coid] - filled, f"{coid}: rest anulat greșit"
            if e.kind in _STATE_OF:
                state = _STATE_OF[e.kind]
            terminal = e.kind in _TERMINAL_KINDS
        out[coid] = OrderSummary(tuple(evs), filled, state)
    return out


def check_snapshot(
    snap: BrokerSnapshot,
    adapter: BrokerAdapter,
    summaries: dict[str, OrderSummary],
    sides: dict[str, str],
) -> None:
    """Snapshot-ul este complet și coerent cu evenimentele livrate."""
    assert snap.complete
    assert snap.environment == adapter.environment
    assert snap.account_id == adapter.account_id
    statuses = {o.client_order_id: o for o in snap.orders}
    assert len(statuses) == len(snap.orders), "ordin duplicat în snapshot"
    assert set(statuses) == set(summaries)
    emitted = {e.broker_exec_id for s in summaries.values() for e in s.events}
    assert emitted <= set(snap.execution_ids)
    assert len(snap.execution_ids) == len(set(snap.execution_ids))
    net: dict[str, Decimal] = defaultdict(lambda: ZERO)
    for coid, summary in summaries.items():
        st = statuses[coid]
        assert st.filled_qty == summary.filled_qty, coid
        assert st.state is summary.last_state, f"{coid}: {st.state} vs {summary.last_state}"
        fills = [e for e in summary.events if e.kind in _FILL_KINDS]
        if fills:
            notional = sum((e.price * e.qty for e in fills if e.price and e.qty), ZERO)
            assert st.avg_fill_price is not None
            assert abs(st.avg_fill_price - notional / summary.filled_qty) < D("1e-9")
            sign = D(1) if sides[coid] == "BUY" else D(-1)
            net[st.instrument] += sign * summary.filled_qty
        else:
            assert st.avg_fill_price is None
    assert snap.positions == {k: v for k, v in net.items() if v != 0}
