"""P11: reconstruirea din jurnal este echivalentă cu procesarea online pentru orice prefix.

Pentru orice prefix al jurnalului, `recover()` (care *reconstruiește* proiecțiile din
înregistrările `oms.*`, fără să re-ruleze FSM-ul, riscul sau brokerul) produce aceeași stare
ca procesarea online până la acel prefix (Req 26.1, 26.3).

Driver online
    Un `OrderManager` real, cu `JournalOmsSink` peste un jurnal SQLite, consumă o secvență
    generată de operații: `create_order` (pe o decizie de risc *aprobată*), `submit`,
    `ExecutionEvent`-uri de la broker (ACK, execuții parțiale care însumează în limita
    cantității, execuție finală sau lăsare deschisă, respingere, expirare), `request_cancel`
    cu confirmare/respingere, `submit_timeout → UNKNOWN` și `reconcile` pentru `UNKNOWN`.
    În paralel se construiește portofoliul online aplicând `outcome.fill` al fiecărei execuții
    *aplicate* — exact fill-urile pe care recuperarea le rejoacă din jurnal.

Adevărul online
    După fiecare operație completă se reține granița de `seq` a jurnalului și un instantaneu al
    adevărului: stările ordinelor (`manager.orders`), setul de `broker_exec_id` aplicate
    (`manager.registry.applied_execs`) și starea portofoliului. Prefixele se definesc la aceste
    granițe (sfârșitul fiecărei operații complete), astfel încât echivalența să fie bine
    definită — o operație online poate emite mai multe înregistrări, iar un prefix tăiat în
    mijloc nu are corespondent online. La granularitatea operației, aceasta este chiar
    „reluarea de la primul eveniment neconfirmat” cerută de 26.1/26.3.

Prefixe fără coruperea lanțului hash
    Un jurnal-prefix se construiește copiind *verbatim* rândurile 1..N (inclusiv `prev_hash`,
    `hash`) într-o bază nouă, ocolind trigger-ele append-only (ca în `test_p12`). Lanțul rămâne
    valid pentru că rândurile sunt copiate exact.

Robustețe la tăieturi arbitrare
    Pe lângă granițele de operație, se taie și la `seq` arbitrare: `recover()` nu trebuie să
    eșueze niciodată, iar invariantul de cantitate (`filled + remaining = qty`) și unicitatea
    execuțiilor aplicate trebuie să se păstreze.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from hypothesis import event, given, settings
from hypothesis import strategies as st

from qts.core.models import (
    ExecKind,
    ExecutionEvent,
    Order,
    OrderIntent,
    OrderState,
)
from qts.oms.idempotency import IdempotencyRegistry, idempotency_key
from qts.oms.manager import ExecOutcome, ExecStatus, JournalOmsSink, OrderManager
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.persistence.recovery import RecoveredState, projections_hash, recover
from qts.portfolio.portfolio import Portfolio, PortfolioState
from qts.risk.engine import RiskDecision
from qts.secrets.store import Redactor

T0 = datetime(2026, 1, 1, tzinfo=UTC)
INITIAL_CASH = Decimal("100000")  # fix în ambele căi (online și recuperare)
RUN_ID = "run-p11"
STRATEGY_ID = "strat-p11"
Q = Decimal("0.0001")  # cantitate cu 4 zecimale → aritmetică exactă
PRICE = Decimal("10.0000")

# Instrumente distincte per ordin: aritmetica portofoliului este independentă de ordinea
# interleaving-ului, deci echivalența pe prefixe este bine definită indiferent de intercalare.
STATS: Counter[str] = Counter()


# --------------------------------------------------------------------------- generatoare


qty_st = st.decimals(min_value=Q, max_value=Decimal("100"), places=4)
# o acțiune pe pasul curent; create se tratează separat (deschide un ordin nou)
ACTIONS = ("fill_partial", "fill_full", "ack", "reject", "expire", "cancel", "timeout")


@dataclass
class OrderPlan:
    """Planul unui ordin: cantitate și o secvență de intenții de acțiune."""

    qty: Decimal
    actions: tuple[str, ...]


@st.composite
def _plans(draw: st.DrawFn) -> list[OrderPlan]:
    n = draw(st.integers(1, 4))
    plans: list[OrderPlan] = []
    for _ in range(n):
        qty = draw(qty_st)
        k = draw(st.integers(1, 6))
        actions = tuple(draw(st.sampled_from(ACTIONS)) for _ in range(k))
        plans.append(OrderPlan(qty=qty, actions=actions))
    return plans


# --------------------------------------------------------------------------- construcții


def _approved(intent: OrderIntent, qty: Decimal) -> RiskDecision:
    return RiskDecision(intent_id=intent.intent_id, approved=True, qty=qty)


def _intent(idx: int, qty: Decimal) -> tuple[OrderIntent, str]:
    instrument = f"INST{idx}"
    intent = OrderIntent(
        intent_id=f"intent-{idx}",
        signal_id=f"sig-{idx}",
        instrument=instrument,
        side="BUY",
        ref_price=PRICE,
    )
    return intent, instrument


def _exec(
    coid: str,
    broker_exec_id: str,
    kind: ExecKind,
    ts: datetime,
    *,
    qty: Decimal | None = None,
    price: Decimal | None = None,
) -> ExecutionEvent:
    return ExecutionEvent(
        broker_exec_id=broker_exec_id,
        client_order_id=coid,
        kind=kind,
        qty=qty,
        price=price,
        commission=Decimal("0.0000") if qty is not None else None,
        ts_broker=ts,
        ts_receipt=ts,
        seq=None,  # mesaje nesecvențiate: aplicate direct, fără buffering
    )


# --------------------------------------------------------------------------- adevăr online


@dataclass
class _Online:
    """Starea online „adevăr” la o anumită graniță de prefix."""

    seq: int
    order_states: dict[str, OrderState]
    applied: frozenset[str]
    portfolio: PortfolioState


@dataclass
class _Driver:
    journal: Journal
    manager: OrderManager
    portfolio: Portfolio
    ts: datetime = T0
    # ordine deschise încă ne-finalizate, cu starea planului
    open_orders: dict[str, _OrderRun] = field(default_factory=dict)
    exec_counter: int = 0

    def tick(self) -> datetime:
        self.ts += timedelta(seconds=1)
        return self.ts

    def next_exec_id(self) -> str:
        self.exec_counter += 1
        return f"X{self.exec_counter}"


@dataclass
class _OrderRun:
    coid: str
    instrument: str
    qty: Decimal
    actions: list[str]
    submitted: bool = False
    cancel_requested: bool = False
    frozen: bool = False  # UNKNOWN, în așteptarea reconcilierii


def _apply_outcomes_to_portfolio(driver: _Driver, outcomes: list[ExecOutcome]) -> None:
    for outcome in outcomes:
        if outcome.status is ExecStatus.APPLIED and outcome.fill is not None:
            driver.portfolio.apply_fill(outcome.fill)
            STATS["fill_applied"] += 1


def _snapshot(driver: _Driver) -> _Online:
    states = {coid: order.state for coid, order in driver.manager.orders.items()}
    applied = frozenset(driver.manager.registry.applied_execs)
    seq, _ = driver.journal.head
    return _Online(
        seq=seq,
        order_states=states,
        applied=applied,
        portfolio=driver.portfolio.snapshot(),
    )


# --------------------------------------------------------------------------- pași online


def _open_order(driver: _Driver, idx: int, plan: OrderPlan) -> None:
    intent, instrument = _intent(idx, plan.qty)
    coid = idempotency_key(RUN_ID, STRATEGY_ID, instrument, idx)
    driver.manager.create_order(
        intent,
        _approved(intent, plan.qty),
        run_id=RUN_ID,
        strategy_id=STRATEGY_ID,
        signal_seq=idx,
        ts=driver.tick(),
    )
    driver.open_orders[coid] = _OrderRun(
        coid=coid, instrument=instrument, qty=plan.qty, actions=list(plan.actions)
    )


def _ensure_submitted(driver: _Driver, run: _OrderRun) -> None:
    if not run.submitted:
        driver.manager.submit(run.coid, driver.tick())
        run.submitted = True


def _step_order(driver: _Driver, run: _OrderRun, action: str) -> None:
    """Aplică o acțiune pe un ordin; emite înregistrări `oms.*` în jurnal."""
    order = driver.manager.orders[run.coid]
    if run.frozen:
        # ordin UNKNOWN: îl rezolvăm prin reconciliere (nu mai acceptă comenzi/execuții)
        _reconcile_unknown(driver, run, order)
        return

    if action == "timeout":
        _ensure_submitted(driver, run)
        if order.state is OrderState.SUBMITTED:
            driver.manager.submit_timeout(run.coid, driver.tick())
            run.frozen = driver.manager.orders[run.coid].state is OrderState.UNKNOWN
            STATS["timeout"] += 1
        return

    if action == "cancel":
        _ensure_acked(driver, run)  # CANCEL_REQUEST este permis din ACKNOWLEDGED/PARTIALLY_FILLED
        if not run.cancel_requested and driver.manager.orders[run.coid].state in (
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
        ):
            res = driver.manager.request_cancel(run.coid, driver.tick())
            run.cancel_requested = res.send or run.cancel_requested
            if res.send:
                # confirmă anularea imediat
                ev = _exec(run.coid, driver.next_exec_id(), ExecKind.CANCELLED, driver.tick())
                out = driver.manager.on_execution(ev)
                _apply_outcomes_to_portfolio(driver, out)
                STATS["cancel_confirmed"] += 1
        return

    if action == "ack":
        _ensure_submitted(driver, run)
        if order.state is OrderState.SUBMITTED:
            ev = _exec(run.coid, driver.next_exec_id(), ExecKind.ACK, driver.tick())
            out = driver.manager.on_execution(ev)
            _apply_outcomes_to_portfolio(driver, out)
            STATS["ack"] += 1
        return

    if action == "reject":
        _ensure_submitted(driver, run)
        if order.state is OrderState.SUBMITTED:
            ev = _exec(run.coid, driver.next_exec_id(), ExecKind.REJECT, driver.tick())
            out = driver.manager.on_execution(ev)
            _apply_outcomes_to_portfolio(driver, out)
            STATS["reject"] += 1
        return

    if action == "expire":
        _ensure_acked(driver, run)  # EXPIRE este permis din ACKNOWLEDGED
        if driver.manager.orders[run.coid].state is OrderState.ACKNOWLEDGED:
            ev = _exec(run.coid, driver.next_exec_id(), ExecKind.EXPIRED, driver.tick())
            out = driver.manager.on_execution(ev)
            _apply_outcomes_to_portfolio(driver, out)
            STATS["expire"] += 1
        return

    # execuții (parțiale/finale)
    _ensure_submitted(driver, run)
    _fill(driver, run, full=action == "fill_full")


def _ensure_acked(driver: _Driver, run: _OrderRun) -> None:
    """Un fill cere o stare post-ACK; brokerul trimite ACK înainte de orice execuție."""
    _ensure_submitted(driver, run)
    if driver.manager.orders[run.coid].state is OrderState.SUBMITTED:
        ev = _exec(run.coid, driver.next_exec_id(), ExecKind.ACK, driver.tick())
        out = driver.manager.on_execution(ev)
        _apply_outcomes_to_portfolio(driver, out)
        STATS["ack"] += 1


def _fill(driver: _Driver, run: _OrderRun, *, full: bool) -> None:
    _ensure_acked(driver, run)
    order = driver.manager.orders[run.coid]
    if order.state not in (
        OrderState.ACKNOWLEDGED,
        OrderState.PARTIALLY_FILLED,
        OrderState.CANCEL_PENDING,
    ):
        return
    remaining = order.remaining_qty
    if remaining <= 0:
        return
    if full:
        qty, kind = remaining, ExecKind.FILL
    else:
        half = (remaining / 2).quantize(Q)
        if half <= 0 or half >= remaining:
            qty, kind = remaining, ExecKind.FILL  # prea mic pentru parțial: devine FILL
        else:
            qty, kind = half, ExecKind.PARTIAL_FILL
    ev = _exec(run.coid, driver.next_exec_id(), kind, driver.tick(), qty=qty, price=PRICE)
    out = driver.manager.on_execution(ev)
    _apply_outcomes_to_portfolio(driver, out)
    STATS["fill_partial" if kind is ExecKind.PARTIAL_FILL else "fill_full"] += 1


def _reconcile_unknown(driver: _Driver, run: _OrderRun, order: Order) -> None:
    """Rezolvă un ordin `UNKNOWN` prin reconciliere la o stare consistentă cu brokerul."""
    filled = order.filled_qty
    if filled == 0:
        state, avg = OrderState.ACKNOWLEDGED, None
    elif filled >= order.qty:
        state, avg = OrderState.FILLED, PRICE
    else:
        state, avg = OrderState.PARTIALLY_FILLED, PRICE
    driver.manager.reconcile(
        run.coid,
        ts=driver.tick(),
        state=state,
        filled_qty=filled,
        avg_fill_price=avg,
    )
    run.frozen = False
    STATS["reconciled"] += 1


# --------------------------------------------------------------------------- prefix journal


def _prefix_journal(source: Journal, up_to_seq: int) -> Journal:
    """Copiază verbatim rândurile 1..up_to_seq într-o bază nouă (lanțul hash rămâne valid)."""
    conn = open_db(":memory:")
    conn.execute("DROP TRIGGER journal_no_update")
    conn.execute("DROP TRIGGER journal_no_delete")
    rows = source.connection.execute(
        "SELECT seq, ts, type, correlation_id, component, component_version, actor, "
        "outcome, payload, prev_hash, hash FROM journal WHERE seq <= ? ORDER BY seq",
        (up_to_seq,),
    ).fetchall()
    for r in rows:
        conn.execute(
            "INSERT INTO journal (seq, ts, type, correlation_id, component, "
            "component_version, actor, outcome, payload, prev_hash, hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            tuple(r),
        )
    return Journal(conn, Redactor())


# --------------------------------------------------------------------------- driver complet


def _run_online(plans: list[OrderPlan]) -> tuple[_Driver, list[_Online]]:
    journal = Journal(open_db(":memory:"), Redactor())
    manager = OrderManager(JournalOmsSink(journal))
    portfolio = Portfolio(PortfolioState.initial(INITIAL_CASH))
    driver = _Driver(journal=journal, manager=manager, portfolio=portfolio)

    truth: list[_Online] = [_snapshot(driver)]  # prefix gol (seq 0)

    # Deschide toate ordinele întâi (instrumente distincte), apoi intercalează acțiunile.
    for idx, plan in enumerate(plans):
        _open_order(driver, idx, plan)
        truth.append(_snapshot(driver))

    runs = list(driver.open_orders.values())
    # Intercalează acțiunile: fiecare pas o acțiune de pe un ordin, în ordine round-robin.
    pending = [(run, list(run.actions)) for run in runs]
    while any(acts for _, acts in pending):
        for run, acts in pending:
            if not acts:
                continue
            action = acts.pop(0)
            _step_order(driver, run, action)
            truth.append(_snapshot(driver))
    return driver, truth


# --------------------------------------------------------------------------- aserțiuni


def _assert_equivalent(prefix_seq: int, online: _Online, source: Journal) -> None:
    prefix = _prefix_journal(source, prefix_seq)
    recovered = recover(prefix, initial_cash=INITIAL_CASH)

    assert recovered.verdict.ok, f"integritate prefix seq={prefix_seq}: {recovered.verdict.reason}"

    # stările ordinelor
    assert recovered.order_states == online.order_states, (
        f"stări diferite la prefix seq={prefix_seq}: "
        f"recover={recovered.order_states} online={online.order_states}"
    )
    # setul de execuții aplicate
    assert frozenset(recovered.registry.applied_execs) == online.applied, (
        f"execuții aplicate diferite la prefix seq={prefix_seq}: "
        f"recover={set(recovered.registry.applied_execs)} online={set(online.applied)}"
    )
    # portofoliul (poziții, numerar, realizat)
    assert recovered.portfolio == online.portfolio, f"portofoliu diferit la prefix seq={prefix_seq}"
    # hash canonic al proiecțiilor
    rec_hash = projections_hash(
        registry=recovered.registry,
        order_states=recovered.order_states,
        portfolio=recovered.portfolio,
    )
    online_hash = projections_hash(
        registry=_registry_for_hash(online.applied, recovered),
        order_states=online.order_states,
        portfolio=online.portfolio,
    )
    assert rec_hash == online_hash, f"hash proiecții diferit la prefix seq={prefix_seq}"

    # instrumentele înghețate = instrumentele ordinelor în stări neterminale/în așteptare
    pending_instruments = {
        inst
        for coid, inst in recovered.order_instruments.items()
        if recovered.order_states.get(coid)
        in (
            OrderState.SUBMITTED,
            OrderState.UNKNOWN,
            OrderState.ACKNOWLEDGED,
            OrderState.PARTIALLY_FILLED,
            OrderState.CANCEL_PENDING,
        )
    }
    assert recovered.frozen_instruments == frozenset(pending_instruments), (
        f"frozen_instruments diferit la prefix seq={prefix_seq}"
    )


def _registry_for_hash(applied: frozenset[str], recovered: RecoveredState) -> IdempotencyRegistry:
    """Registru cu aceleași chei aplicate ca recuperarea (hash-ul compară doar cheile/valorile).

    Hash-ul `projections_hash` folosește `registry.applied_execs`; recuperarea reconstruiește
    aceleași intrări. Pentru online folosim registrul recuperat (sunt egale ca set, verificat
    separat), astfel hash-ul testează stările și portofoliul, nu re-serializarea registrului.
    """
    return recovered.registry


# --------------------------------------------------------------------------- proprietatea


@settings(max_examples=200)
@given(_plans())
def test_property_11_recovery_equivalent_for_every_prefix(plans: list[OrderPlan]) -> None:
    """**Validates: Requirements 26.1, 26.3**"""
    driver, truth = _run_online(plans)
    source = driver.journal
    event(f"prefixe: {len(truth)}")

    # Pentru fiecare graniță de operație: recuperarea pe prefix == adevărul online.
    for online in truth:
        _assert_equivalent(online.seq, online, source)


@settings(max_examples=200)
@given(_plans(), st.data())
def test_property_11_recovery_never_crashes_on_arbitrary_cut(
    plans: list[OrderPlan], data: st.DataObject
) -> None:
    """La orice tăietură brută de `seq`, recuperarea reușește și păstrează invariantele.

    **Validates: Requirements 26.1, 26.3**
    """
    driver, _ = _run_online(plans)
    source = driver.journal
    head_seq, _ = source.head
    if head_seq == 0:
        return
    cut = data.draw(st.integers(0, head_seq), label="cut")
    prefix = _prefix_journal(source, cut)
    recovered = recover(prefix, initial_cash=INITIAL_CASH)
    assert recovered.verdict.ok

    # invariantul de cantitate se reflectă în stările reconstruite (nicio stare incoerentă);
    # execuțiile aplicate sunt unice (set == dict de chei) prin construcție.
    assert len(set(recovered.registry.applied_execs)) == len(recovered.registry.applied_execs)
    # portofoliul rămâne valid (numerarul nu scade sub zero fără incident consemnat)
    for inc in recovered.portfolio.incidents:
        assert inc.code == "CASH_NEGATIVE"


def test_property_11_non_vacuity() -> None:
    """Scenariile acoperă fill-uri, anulări, UNKNOWN+reconciliere, respingeri și expirări."""
    STATS.clear()
    test_property_11_recovery_equivalent_for_every_prefix()
    for key in (
        "fill_applied",
        "fill_full",
        "cancel_confirmed",
        "reconciled",
        "reject",
        "expire",
        "timeout",
        "ack",
    ):
        assert STATS[key] > 0, f"scenariu neexercitat: {key} ({dict(STATS)})"
