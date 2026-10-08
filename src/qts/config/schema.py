"""Schema versionată a configurației (Req 17.1, 17.2, 17.6).

Câmpurile necunoscute sunt refuzate. Valorile de risc sunt mărginite de limitele aprobate în
cerințe: niciun fișier de configurație nu poate depăși 0,50 EUR per tranzacție, 2 EUR pe zi sau
10 EUR pierdere totală (Req 13, 28.7). Secretele apar numai ca referințe (`secret_ref`).
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Final, Literal, get_args

from pydantic import Field, field_validator, model_validator

from qts.core.models import Dec, Frozen, Instrument
from qts.costs.model import CostModelConfig

SCHEMA_VERSION: Final = "1"

Environment = Literal["backtest", "shadow", "demo", "live"]
ENVIRONMENTS: Final[tuple[str, ...]] = get_args(Environment)
BrokerKind = Literal["sim", "fake", "demo", "live"]

# O referință la secret are forma `qts/<mediu>/<nume>`: fără spații, fără valoarea secretului.
SECRET_REF_PATTERN: Final = re.compile(r"^qts/[a-z0-9_-]+(/[a-z0-9_.-]+)+$")


class ConfigIssuesError(ValueError):
    """Mai multe abateri detectate de același validator; loader-ul le raportează separat."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = list(issues)
        super().__init__("; ".join(self.issues))


MAX_RISK_PER_TRADE_EUR: Final = Decimal("0.50")
MIN_RISK_PER_TRADE_TARGET_EUR: Final = Decimal("0.25")
MAX_DAILY_LOSS_EUR: Final = Decimal("2")
MAX_TOTAL_LOSS_EUR: Final = Decimal("10")
REFERENCE_CAPITAL_EUR: Final = Decimal("100")

# Brokerul permis pentru fiecare mod (Req 1.2).
ALLOWED_BROKER_BY_ENV: Final[dict[str, frozenset[str]]] = {
    "backtest": frozenset({"sim"}),
    "shadow": frozenset({"sim"}),
    "demo": frozenset({"demo", "fake"}),
    "live": frozenset({"live"}),
}


class RunConfig(Frozen):
    seed: int = Field(ge=0)
    db_path: str
    label: str = ""


class DataConfig(Frozen):
    source_id: str
    dataset_path: str | None = None
    bar_interval_min: int = Field(ge=5, le=60)  # Req 4.5
    default_freshness_seconds: int = Field(gt=0)
    freshness_seconds: dict[str, int] = Field(default_factory=dict)
    max_gap_bars: int = Field(default=3, ge=0)  # Req 7.4, 7.7

    @model_validator(mode="after")
    def _positive_thresholds(self) -> DataConfig:
        bad = [k for k, v in self.freshness_seconds.items() if v <= 0]
        if bad:
            raise ValueError(f"praguri de prospețime ≤ 0 pentru: {sorted(bad)}")
        return self


class BrokerConfig(Frozen):
    kind: BrokerKind
    name: str | None = None
    endpoint: str | None = None
    account_id: str | None = None
    secret_ref: str | None = None

    @field_validator("secret_ref")
    @classmethod
    def _is_reference(cls, value: str | None) -> str | None:
        # Valoarea nu este afișată în eroare: ar putea fi chiar secretul.
        if value is not None and not SECRET_REF_PATTERN.fullmatch(value):
            raise ValueError("secret_ref trebuie să fie o referință de forma qts/<mediu>/<nume>")
        return value


class RiskConfig(Frozen):
    reference_capital_eur: Dec = REFERENCE_CAPITAL_EUR
    risk_per_trade_target_eur: Dec = Decimal("0.25")
    risk_per_trade_max_eur: Dec = MAX_RISK_PER_TRADE_EUR
    daily_loss_limit_eur: Dec = MAX_DAILY_LOSS_EUR
    total_loss_limit_eur: Dec = MAX_TOTAL_LOSS_EUR
    max_open_positions: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _within_approved_bounds(self) -> RiskConfig:
        errors: list[str] = []
        if self.reference_capital_eur != REFERENCE_CAPITAL_EUR:
            errors.append("reference_capital_eur trebuie să fie 100 (Req 28.1)")
        if not (
            MIN_RISK_PER_TRADE_TARGET_EUR
            <= self.risk_per_trade_target_eur
            <= self.risk_per_trade_max_eur
        ):
            errors.append("risk_per_trade_target_eur trebuie în [0.25, risk_per_trade_max_eur]")
        if not Decimal(0) < self.risk_per_trade_max_eur <= MAX_RISK_PER_TRADE_EUR:
            errors.append("risk_per_trade_max_eur trebuie în (0, 0.50] (Req 13.2)")
        if not Decimal(0) < self.daily_loss_limit_eur <= MAX_DAILY_LOSS_EUR:
            errors.append("daily_loss_limit_eur trebuie în (0, 2] (Req 13.3)")
        if self.total_loss_limit_eur != MAX_TOTAL_LOSS_EUR:
            errors.append("total_loss_limit_eur trebuie să fie 10 (Req 13.5, 28.7)")
        if errors:
            raise ConfigIssuesError(errors)
        return self


class StrategyConfig(Frozen):
    strategy_id: str
    params: dict[str, str | int] = Field(default_factory=dict)


class KillSwitchConfig(Frozen):
    open_orders_policy: Literal["keep", "cancel"] = "keep"  # Req 14.7


class AppConfig(Frozen):
    schema_version: Literal["1"]
    environment: Environment
    run: RunConfig
    data: DataConfig
    broker: BrokerConfig
    risk: RiskConfig = RiskConfig()
    strategy: StrategyConfig
    kill_switch: KillSwitchConfig = KillSwitchConfig()
    instruments: list[Instrument] = Field(min_length=1)
    # Complete_Cost_Model versionat (Req 8.1-8.3). Opțional în schemă; modurile care folosesc
    # `SimBroker` (Backtest, Shadow) refuză pornirea fără el (`bootstrap.py`).
    costs: CostModelConfig | None = None

    @model_validator(mode="after")
    def _mode_consistency(self) -> AppConfig:
        errors: list[str] = []
        allowed = ALLOWED_BROKER_BY_ENV[self.environment]
        if self.broker.kind not in allowed:
            errors.append(
                f"broker.kind={self.broker.kind} nu este permis în {self.environment}; "
                f"permis: {sorted(allowed)}"
            )
        if self.broker.kind in ("demo", "live"):
            for field in ("name", "endpoint", "account_id", "secret_ref"):
                if getattr(self.broker, field) is None:
                    errors.append(f"broker.{field} lipsă pentru broker.kind={self.broker.kind}")
        if self.environment == "backtest" and self.data.dataset_path is None:
            errors.append("data.dataset_path lipsă pentru backtest")
        symbols = [i.symbol for i in self.instruments]
        if len(symbols) != len(set(symbols)):
            errors.append("instruments conține simboluri duplicate")
        if errors:
            raise ConfigIssuesError(errors)
        return self
