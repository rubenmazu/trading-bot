"""Fabrica de adaptoare broker (Req 1.2, 2.1, 2.2; design: Live_Gate și Fail_Safe_Block).

Singurul loc care transformă o `AppConfig` validată într-un adaptor concret. Pași, în ordine:

1. verificările de pornire din `safety/stage.py` (`startup_violations`): modul Live, brokerul
   `live`, endpoint-urile și conturile live cunoscute sunt refuzate *înaintea* construirii
   oricărui adaptor. Pentru Live (mod sau `broker.kind`) se ridică eroarea dedicată
   `LiveBrokerRefusedError`, indiferent de etapă: adaptoarele live aparțin pachetului opțional
   `qts_live`, pe care această fabrică nu îl importă și nu îl referă (bariera 1);
2. selecția adaptorului numai din mediul ales (1.2):
   - `sim` (backtest, shadow) → `SimBroker`;
   - `fake` (demo cu endpoint `fake://`) → `FakeBroker` în mediul `demo`;
   - `demo` (broker demo real) → refuzat cu `BrokerNotAvailableError`: adaptorul depinde de
     Open_Decision pentru broker și nu există încă;
3. învelirea fiecărui adaptor în `FailSafeBlock` (bariera 3). Fabrica nu întoarce niciodată un
   adaptor neînvelit.

Maparea mediilor: `ApprovedTarget.environment` păstrează modul din configurație
(backtest/shadow/demo/live), iar `FailSafeBlock` îl compară cu mediul raportat de adaptor prin
`ADAPTER_ENVIRONMENT_BY_MODE` (backtest/shadow → `sim`). Etapa se verifică pe modul din
configurație, cu aceleași nume ca `ALLOWED_ENVIRONMENTS_BY_STAGE`.

Conturi: pentru `sim` și `fake`, `broker.account_id` este opțional; lipsa lui înseamnă contul
local implicit (`SIM_DEFAULT_ACCOUNT`, `FAKE_DEFAULT_ACCOUNT`), care devine și contul aprobat.

Credențiale: `sim` și `fake` nu folosesc secrete, deci `secret_ref` nu este rezolvat. Un viitor
adaptor demo îl va rezolva prin `SecretStore` numai aici, fără a-l jurnaliza.
"""

from __future__ import annotations

from typing import Final

from qts.config.schema import AppConfig
from qts.core.clock import Clock
from qts.costs.model import CompleteCostModel
from qts.safety.live_gate import LiveGate
from qts.safety.stage import StageInfo, StartupRefusedError, startup_violations

from .adapter import OrderRequest, SubmitAck
from .fail_safe import (
    ApprovedTarget,
    AuditAppender,
    FailSafeBlock,
    LiveGateLike,
    OrderBlocker,
)
from .fake import FakeBroker
from .sim import SimBroker, SimBrokerConfig

__all__ = [
    "FAKE_DEFAULT_ACCOUNT",
    "FAKE_DEFAULT_ENDPOINT",
    "SIM_DEFAULT_ACCOUNT",
    "BrokerFactoryError",
    "BrokerNotAvailableError",
    "GuardedAdapter",
    "LiveBrokerRefusedError",
    "build_broker",
]

SIM_DEFAULT_ACCOUNT: Final = "SIM-LOCAL"
FAKE_DEFAULT_ACCOUNT: Final = "FAKE-DEMO-1"
FAKE_DEFAULT_ENDPOINT: Final = "fake://local"

GuardedAdapter = FailSafeBlock[OrderRequest, SubmitAck]


class BrokerFactoryError(Exception):
    """Fabrica nu poate construi adaptorul cerut."""


class BrokerNotAvailableError(BrokerFactoryError):
    """Adaptorul cerut nu există în această versiune."""


class LiveBrokerRefusedError(StartupRefusedError):
    """Modul Live sau brokerul `live` au fost ceruți; refuzați în această versiune (Req 2.2)."""


def _refuse_before_construction(config: AppConfig, stage: StageInfo) -> None:
    reasons = startup_violations(config, stage)
    if config.environment == "live" or config.broker.kind == "live":
        # Adaptoarele live (pachetul `qts_live`) nu sunt disponibile; refuz indiferent de etapă.
        live_reason = "adaptoarele Live nu sunt disponibile în această versiune (Req 2.2)"
        raise LiveBrokerRefusedError([*reasons, live_reason])
    if reasons:
        raise StartupRefusedError(reasons)


def build_broker(
    config: AppConfig,
    stage: StageInfo,
    *,
    clock: Clock,
    kill_switch: OrderBlocker,
    audit: AuditAppender,
    cost_model: CompleteCostModel | None = None,
    sim_config: SimBrokerConfig | None = None,
    live_gate: LiveGateLike | None = None,
) -> GuardedAdapter:
    """Construiește adaptorul modului ales, învelit în `FailSafeBlock`.

    `cost_model` este obligatoriu pentru `sim`. `live_gate` implicit: `LiveGate` evaluat pe
    etapa dată, deci închis în Initial_Stage.
    """
    _refuse_before_construction(config, stage)
    broker = config.broker
    adapter: SimBroker | FakeBroker
    if broker.kind == "sim":
        if cost_model is None:
            raise BrokerFactoryError("broker.kind=sim necesită un model de costuri complet")
        account = broker.account_id or SIM_DEFAULT_ACCOUNT
        adapter = SimBroker(
            account_id=account,
            instruments=config.instruments,
            cost_model=cost_model,
            clock=clock,
            config=sim_config,
        )
    elif broker.kind == "fake":
        endpoint = broker.endpoint or FAKE_DEFAULT_ENDPOINT
        if not endpoint.lower().startswith("fake://"):
            raise BrokerFactoryError("broker.kind=fake necesită un endpoint fake://")
        account = broker.account_id or FAKE_DEFAULT_ACCOUNT
        adapter = FakeBroker(
            instruments=config.instruments,
            clock=clock,
            account_id=account,
            environment="demo",
            endpoint=endpoint,
        )
    elif broker.kind == "demo":
        raise BrokerNotAvailableError(
            "broker demo adapter not implemented; Open_Decision pending "
            "(adaptorul demo depinde de alegerea brokerului)"
        )
    else:  # pragma: no cover - `live` este refuzat mai sus; schema nu permite alte valori
        raise BrokerFactoryError(f"broker.kind={broker.kind!r} nu este suportat")

    approved = ApprovedTarget(environment=config.environment, account_id=account)
    return FailSafeBlock(
        adapter,
        approved=approved,
        stage=stage,
        kill_switch=kill_switch,
        audit=audit,
        clock=clock,
        live_gate=live_gate if live_gate is not None else LiveGate(stage=lambda: stage),
    )
