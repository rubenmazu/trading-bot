"""Strategia de referință: mean-reversion pe bare de 15 minute (Req 6.3, 6.4).

Reguli (versiunea `1.0.0`), evaluate la închiderea fiecărei bare:

- `mean` și `std` sunt media și abaterea standard **de populație** a ultimelor `lookback`
  prețuri de închidere vizibile, inclusiv bara curentă; `z = (close - mean) / std`.
- Fără poziție: `z <= -entry_z` → `ENTER_LONG` cu `stop = floor(close - stop_k × std)` la
  `price_quantum` (rotunjirea în jos mărește distanța până la stop, deci nu reduce riscul
  estimat de `Risk_Engine`).
- Cu poziție: întâi stopul, apoi revenirea la medie.
  * `low <= stop` → `EXIT` (`EXIT_STOP`). Se folosește minimul barei, nu închiderea, pentru că
    o atingere intrabară a stopului trebuie tratată ca stop declanșat (alegere conservatoare).
  * `z >= exit_z` → `EXIT` (`EXIT_MEAN`); implicit `exit_z = 0`, adică închiderea a revenit
    la medie.
- Altfel `NONE` cu `NO_RULE_MATCHED`.

Intrările absente sau invalide (interval diferit de 15 minute, bară incoerentă, istoric
insuficient sau incoerent, volatilitate zero) produc `NONE` cu un cod de motiv specific.

Strategia nu stabilește cantitatea. Starea urmărește poziția *intenționată* de semnalele
strategiei; reconcilierea cu execuțiile reale aparține motorului. Toate calculele folosesc
`Decimal` într-un context local (precizie 28, `ROUND_HALF_EVEN`), deci rezultatul este
determinist.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from decimal import ROUND_FLOOR, ROUND_HALF_EVEN, Context, Decimal, localcontext
from typing import Annotated, Final, Literal

from pydantic import Field, model_validator

from qts.config.schema import StrategyConfig
from qts.core.models import Bar, Dec, Frozen, Signal
from qts.strategy.base import (
    REASON_INSUFFICIENT_HISTORY,
    REASON_INVALID_INPUT,
    REASON_NO_RULE_MATCHED,
    StrategyState,
    bar_data_id,
    build_signal,
    none_signal,
)
from qts.strategy.history_view import HistoryView

STRATEGY_ID: Final = "mean_reversion_v1"
STRATEGY_VERSION: Final = "1.0.0"
INTERVAL_MIN: Final = 15

# Coduri de motiv specifice (Req 6.3, 6.4).
REASON_ENTRY_ZSCORE: Final = "ENTRY_ZSCORE"
REASON_EXIT_MEAN: Final = "EXIT_MEAN"
REASON_EXIT_STOP: Final = "EXIT_STOP"
REASON_INVALID_INTERVAL: Final = "INVALID_INTERVAL"
REASON_INVALID_BAR: Final = "INVALID_BAR"
REASON_HISTORY_MISMATCH: Final = "HISTORY_MISMATCH"
REASON_ZERO_VOLATILITY: Final = "ZERO_VOLATILITY"
REASON_INVALID_STOP: Final = "INVALID_STOP"

_CTX: Final = Context(prec=28, rounding=ROUND_HALF_EVEN)
_ZERO: Final = Decimal(0)


class MeanReversionParams(Frozen):
    """Parametrii versionați ai strategiei, citiți din `StrategyConfig.params`."""

    lookback: Annotated[int, Field(ge=2, le=1000)] = 20
    entry_z: Dec = Decimal("2")
    exit_z: Dec = Decimal("0")
    stop_k: Dec = Decimal("3")
    price_quantum: Dec = Decimal("0.0001")

    @model_validator(mode="after")
    def _check(self) -> MeanReversionParams:
        if self.entry_z <= 0:
            raise ValueError("entry_z trebuie să fie > 0")
        if self.exit_z <= -self.entry_z:
            raise ValueError("exit_z trebuie să fie > -entry_z")
        if self.stop_k <= 0:
            raise ValueError("stop_k trebuie să fie > 0")
        if self.price_quantum <= 0:
            raise ValueError("price_quantum trebuie să fie > 0")
        return self

    @classmethod
    def from_mapping(cls, params: Mapping[str, str | int]) -> MeanReversionParams:
        """Validează parametrii din configurație; cheile necunoscute sunt respinse."""
        return cls.model_validate(dict(params))

    def as_inputs(self) -> dict[str, Decimal]:
        return {
            "param.lookback": Decimal(self.lookback),
            "param.entry_z": self.entry_z,
            "param.exit_z": self.exit_z,
            "param.stop_k": self.stop_k,
            "param.price_quantum": self.price_quantum,
        }


class MeanReversionState(StrategyState):
    in_position: bool = False
    entry_price: Dec | None = None
    stop_price: Dec | None = None

    @model_validator(mode="after")
    def _consistent(self) -> MeanReversionState:
        has_levels = self.entry_price is not None and self.stop_price is not None
        if self.in_position != has_levels:
            raise ValueError("in_position necesită entry_price și stop_price (și invers)")
        if self.entry_price is None and self.stop_price is not None:
            raise ValueError("stop_price fără entry_price")
        return self


FLAT: Final = MeanReversionState()


def _bar_issue(bar: Bar) -> str | None:
    """Codul de motiv pentru o bară inutilizabilă sau `None` dacă bara este validă."""
    if bar.interval_min != INTERVAL_MIN or bar.ts_close - bar.ts_open != timedelta(
        minutes=INTERVAL_MIN
    ):
        return REASON_INVALID_INTERVAL
    if (
        bar.low <= 0
        or bar.volume < 0
        or bar.low > min(bar.open, bar.close)
        or bar.high < max(bar.open, bar.close)
    ):
        return REASON_INVALID_BAR
    return None


def _mean_std(values: Sequence[Decimal]) -> tuple[Decimal, Decimal]:
    """Media și abaterea standard de populație, în context Decimal local."""
    with localcontext(_CTX):
        n = Decimal(len(values))
        mean = sum(values, _ZERO) / n
        var = sum(((v - mean) * (v - mean) for v in values), _ZERO) / n
        return mean, var.sqrt()


class MeanReversionStrategy:
    """Mean-reversion long-only pe 15 minute; satisface protocolul `Strategy`.

    `config_snapshot_id` se fixează la construcție: o instanță aparține unui singur
    `Configuration_Snapshot`, iar fiecare `Signal` emis îl poartă (Req 6.5).
    """

    strategy_id: str = STRATEGY_ID
    version: str = STRATEGY_VERSION

    def __init__(self, params: MeanReversionParams, *, config_snapshot_id: str) -> None:
        if not config_snapshot_id:
            raise ValueError("config_snapshot_id este obligatoriu")
        self.params = params
        self.config_snapshot_id = config_snapshot_id

    @classmethod
    def from_config(
        cls, config: StrategyConfig, *, config_snapshot_id: str
    ) -> MeanReversionStrategy:
        if config.strategy_id != STRATEGY_ID:
            raise ValueError(f"strategy_id {config.strategy_id!r} diferă de {STRATEGY_ID!r}")
        return cls(
            MeanReversionParams.from_mapping(config.params), config_snapshot_id=config_snapshot_id
        )

    def initial_state(self) -> StrategyState:
        return FLAT

    # ------------------------------------------------------------------ evaluare

    def on_bar(
        self, bar: Bar, view: HistoryView, state: StrategyState
    ) -> tuple[Signal, StrategyState]:
        if not isinstance(state, MeanReversionState):
            raise TypeError(f"stare neașteptată: {type(state).__name__}")
        p = self.params
        inputs: dict[str, Decimal] = p.as_inputs()
        inputs["close"] = bar.close
        # Validatorul stării garantează: in_position ⇔ entry_price și stop_price prezente.
        held_stop = state.stop_price if state.in_position else None
        if state.entry_price is not None and held_stop is not None:
            inputs["entry_price"] = state.entry_price
            inputs["stop"] = held_stop
        rules: list[str] = []

        def none(
            reason: str, data_ids: Sequence[str] | None = None
        ) -> tuple[Signal, StrategyState]:
            sig = none_signal(
                self,
                bar,
                reason_code=reason,
                config_snapshot_id=self.config_snapshot_id,
                inputs=inputs,
                rules_evaluated=rules,
                data_ids=data_ids,
            )
            return sig, state

        # 1. Bara curentă.
        issue = _bar_issue(bar)
        rules.append(f"bar_valid:{issue is None}")
        if issue is not None:
            return none(issue)

        # 2. Coerența vederii cu bara curentă.
        if bar.instrument not in view.instruments() or view.current(bar.instrument) != bar:
            rules.append("history_matches_bar:False")
            return none(REASON_HISTORY_MISMATCH)
        rules.append("history_matches_bar:True")

        # 3. Stopul nu depinde de istoric, deci este evaluat primul când există poziție.
        if held_stop is not None:
            hit = bar.low <= held_stop
            inputs["low"] = bar.low
            rules.append(f"stop_hit(low<=stop):{hit}")
            if hit:
                return self._signal(bar, "EXIT", REASON_EXIT_STOP, inputs, rules), FLAT

        # 4. Fereastra statistică.
        window = view.last(bar.instrument, p.lookback)
        data_ids = [bar_data_id(b) for b in window]
        enough = len(window) == p.lookback
        rules.append(f"history>=lookback:{enough}")
        if not enough:
            return none(REASON_INSUFFICIENT_HISTORY, data_ids)
        window_ok = all(_bar_issue(b) is None for b in window)
        rules.append(f"history_valid:{window_ok}")
        if not window_ok:
            return none(REASON_INVALID_INPUT, data_ids)

        mean, std = _mean_std([b.close for b in window])
        inputs["mean"] = mean
        inputs["std"] = std
        rules.append(f"std>0:{std > 0}")
        if std <= 0:
            return none(REASON_ZERO_VOLATILITY, data_ids)
        with localcontext(_CTX):
            z = (bar.close - mean) / std
        inputs["z"] = z

        # 5. Ieșire la medie.
        if state.in_position:
            reverted = z >= p.exit_z
            rules.append(f"z>=exit_z:{reverted}")
            if reverted:
                return (
                    self._signal(bar, "EXIT", REASON_EXIT_MEAN, inputs, rules, data_ids),
                    FLAT,
                )
            return none(REASON_NO_RULE_MATCHED, data_ids)

        # 6. Intrare.
        entry = z <= -p.entry_z
        rules.append(f"z<=-entry_z:{entry}")
        if not entry:
            return none(REASON_NO_RULE_MATCHED, data_ids)
        with localcontext(_CTX):
            raw_stop = bar.close - p.stop_k * std
            stop = (raw_stop / p.price_quantum).to_integral_value(rounding=ROUND_FLOOR) * (
                p.price_quantum
            )
        inputs["stop"] = stop
        stop_ok = _ZERO < stop < bar.close
        rules.append(f"0<stop<close:{stop_ok}")
        if not stop_ok:
            return none(REASON_INVALID_STOP, data_ids)
        signal = self._signal(
            bar, "ENTER_LONG", REASON_ENTRY_ZSCORE, inputs, rules, data_ids, stop_price=stop
        )
        new_state = MeanReversionState(in_position=True, entry_price=bar.close, stop_price=stop)
        return signal, new_state

    def _signal(
        self,
        bar: Bar,
        action: Literal["ENTER_LONG", "EXIT"],
        reason: str,
        inputs: Mapping[str, Decimal],
        rules: Sequence[str],
        data_ids: Sequence[str] | None = None,
        *,
        stop_price: Decimal | None = None,
    ) -> Signal:
        return build_signal(
            self,
            bar,
            action=action,
            reason_code=reason,
            config_snapshot_id=self.config_snapshot_id,
            inputs=inputs,
            rules_evaluated=rules,
            data_ids=data_ids,
            stop_price=stop_price,
        )
