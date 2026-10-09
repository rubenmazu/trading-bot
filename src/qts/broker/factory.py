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
   - `demo` cu `broker.name == "alpaca"` → `AlpacaBrokerAdapter` (paper), cu cheile rezolvate din
     `SecretStore`; orice alt broker demo este refuzat cu `BrokerNotAvailableError`;
3. învelirea fiecărui adaptor în `FailSafeBlock` (bariera 3). Fabrica nu întoarce niciodată un
   adaptor neînvelit.

Maparea mediilor: `ApprovedTarget.environment` păstrează modul din configurație
(backtest/shadow/demo/live), iar `FailSafeBlock` îl compară cu mediul raportat de adaptor prin
`ADAPTER_ENVIRONMENT_BY_MODE` (backtest/shadow → `sim`). Etapa se verifică pe modul din
configurație, cu aceleași nume ca `ALLOWED_ENVIRONMENTS_BY_STAGE`.

Conturi: pentru `sim` și `fake`, `broker.account_id` este opțional; lipsa lui înseamnă contul
local implicit (`SIM_DEFAULT_ACCOUNT`, `FAKE_DEFAULT_ACCOUNT`), care devine și contul aprobat.

Credențiale: `sim` și `fake` nu folosesc secrete, deci `secret_ref` nu este rezolvat. Brokerul
demo real (Alpaca paper) le rezolvă prin `SecretStore` numai aici, fără a le jurnaliza: din
`broker.secret_ref` se derivă referința cheii API și, prin sufixul `_secret`, referința
secretului.
"""

from __future__ import annotations

from typing import Final

from qts.config.schema import AppConfig
from qts.core.clock import Clock
from qts.costs.model import CompleteCostModel
from qts.safety.live_gate import LiveGate
from qts.safety.stage import StageInfo, StartupRefusedError, startup_violations
from qts.secrets.store import Identity, SecretRef, SecretStore

from .adapter import OrderRequest, SubmitAck
from .alpaca_broker import (
    ALPACA_PAPER_ENDPOINT,
    AlpacaBrokerAdapter,
    AlpacaBrokerCredentials,
    AlpacaClientFactory,
)
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


# Sufixul referinței secretului Alpaca (nu este un secret, ci o convenție de denumire a ref-ului).
ALPACA_SECRET_SUFFIX: Final = "_secret"  # noqa: S105


def _alpaca_credentials(secret_ref: str) -> AlpacaBrokerCredentials:
    """Derivă cele două referințe Alpaca din `broker.secret_ref` (convenție de nume).

    `secret_ref` este referința cheii API (`qts/<mediu>/<nume>`); referința secretului este
    aceeași cu sufixul `_secret`. Astfel configurația conține o singură referință, iar ambele
    chei rămân în Secret_Store, niciodată în config.
    """
    return AlpacaBrokerCredentials(
        api_key_ref=SecretRef(secret_ref),
        api_secret_ref=SecretRef(f"{secret_ref}{ALPACA_SECRET_SUFFIX}"),
    )


def _build_alpaca_demo(
    config: AppConfig,
    *,
    clock: Clock,
    secret_store: SecretStore | None,
    client_factory: AlpacaClientFactory | None,
) -> AlpacaBrokerAdapter:
    """Construiește adaptorul Alpaca paper din configurația demo, cu cheile din Secret_Store."""
    broker = config.broker
    if secret_store is None:
        raise BrokerFactoryError(
            "broker demo 'alpaca' necesită un Secret_Store pentru rezolvarea cheilor API; "
            "transmiteți secret_store la build_broker"
        )
    if broker.secret_ref is None:  # pragma: no cover - garantat de schema demo
        raise BrokerFactoryError("broker.secret_ref lipsă pentru broker demo 'alpaca'")
    if broker.account_id is None:  # pragma: no cover - garantat de schema demo
        raise BrokerFactoryError("broker.account_id lipsă pentru broker demo 'alpaca'")
    endpoint = broker.endpoint or ALPACA_PAPER_ENDPOINT
    return AlpacaBrokerAdapter.from_secret_store(
        account_id=broker.account_id,
        instruments=config.instruments,
        store=secret_store,
        credentials=_alpaca_credentials(broker.secret_ref),
        requester=Identity(f"demo-runner:{config.environment}"),
        clock=clock,
        environment=config.environment,
        endpoint=endpoint,
        client_factory=client_factory,
    )


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
    secret_store: SecretStore | None = None,
    alpaca_client_factory: AlpacaClientFactory | None = None,
    reporting_currency: str = "EUR",
) -> GuardedAdapter:
    """Construiește adaptorul modului ales, învelit în `FailSafeBlock`.

    `cost_model` este obligatoriu pentru `sim`. `secret_store` este obligatoriu pentru brokerul
    demo real (Alpaca), care rezolvă cheile API din Secret_Store; `alpaca_client_factory` permite
    injectarea unui client fals în teste (implicit se importă `alpaca-py` la rulare). `live_gate`
    implicit: `LiveGate` evaluat pe etapa dată, deci închis în Initial_Stage. `reporting_currency`
    (implicit EUR) este moneda de raportare a rulării, folosită de `SimBroker` pentru deciziile de
    conversie FX (un instrument în moneda de raportare nu se convertește).
    """
    _refuse_before_construction(config, stage)
    broker = config.broker
    adapter: SimBroker | FakeBroker | AlpacaBrokerAdapter
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
            reporting_currency=reporting_currency,
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
        if (broker.name or "").lower() != "alpaca":
            raise BrokerNotAvailableError(
                f"broker demo {broker.name!r} nu este implementat; singurul broker demo "
                "disponibil este 'alpaca' (paper)"
            )
        adapter = _build_alpaca_demo(
            config,
            clock=clock,
            secret_store=secret_store,
            client_factory=alpaca_client_factory,
        )
        account = adapter.account_id
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
