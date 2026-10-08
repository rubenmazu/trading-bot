"""Validare walk-forward: minimum cinci ferestre Out_Of_Sample_Set ulterioare celor de calibrare.

Req 19.1, 19.3 (indirect, prin fixarea prealabilă), 19.7.

O `Walk_Forward_Validation` împarte axa temporală a datelor (bare indexate cronologic) într-o
secvență de ferestre: fiecare fereastră are un segment `Development_Set` (calibrare/selecție)
urmat *temporal* de un segment `Out_Of_Sample_Set` (evaluare). Lungimea segmentelor
(`train_bars`, `test_bars`), pasul (`step_bars`) și regula de recalibrare (`recalibrate`) sunt
fixate în `WalkForwardSpec` *înaintea* evaluării finale (Req 19.2, impus de pre-înregistrare);
acest modul doar materializează ferestrele conform specificației și verifică invariantele:

- fiecare segment OOS urmează imediat segmentului Development asociat (fără suprapunere temporală);
- se produc cel puțin cinci ferestre OOS (Req 19.1); dacă datele disponibile nu permit cinci
  ferestre complete, construcția ridică `InsufficientDataError` (fail-closed), nu reduce tăcut
  numărul de ferestre;
- ferestrele sunt deterministe: aceeași specificație și același număr de bare produc exact
  aceleași limite, deci rezultatele sunt reproductibile.
"""

from __future__ import annotations

from typing import Annotated, Final

from pydantic import Field, model_validator

from qts.core.models import Frozen
from qts.research.preregistration import WalkForwardSpec

__all__ = [
    "MIN_OOS_WINDOWS",
    "InsufficientDataError",
    "WalkForwardWindow",
    "build_walk_forward_windows",
    "min_bars_required",
]

MIN_OOS_WINDOWS: Final = 5


class InsufficientDataError(Exception):
    """Datele disponibile nu permit minimum cinci ferestre OOS complete (Req 19.1, fail-closed)."""


class WalkForwardWindow(Frozen):
    """O fereastră walk-forward: un segment Development urmat temporal de un segment OOS.

    Indicii sunt poziții de bară semi-deschise `[start, end)` în secvența cronologică de date.
    Invariantul cheie (Req 19.1): `oos_start == dev_end`, deci OOS urmează imediat Development,
    fără suprapunere și fără interval liber.
    """

    index: Annotated[int, Field(ge=0)]
    dev_start: Annotated[int, Field(ge=0)]
    dev_end: Annotated[int, Field(ge=0)]
    oos_start: Annotated[int, Field(ge=0)]
    oos_end: Annotated[int, Field(ge=0)]
    recalibrate: bool

    @model_validator(mode="after")
    def _check(self) -> WalkForwardWindow:
        if self.dev_end <= self.dev_start:
            raise ValueError("segmentul Development trebuie să conțină cel puțin o bară")
        if self.oos_end <= self.oos_start:
            raise ValueError("segmentul Out_Of_Sample trebuie să conțină cel puțin o bară")
        # Req 19.1: OOS urmează imediat Development-ul asociat (ulterioritate temporală strictă).
        if self.oos_start != self.dev_end:
            raise ValueError("segmentul OOS trebuie să urmeze imediat segmentul Development")
        return self

    @property
    def dev_length(self) -> int:
        return self.dev_end - self.dev_start

    @property
    def oos_length(self) -> int:
        return self.oos_end - self.oos_start


def min_bars_required(spec: WalkForwardSpec) -> int:
    """Numărul minim de bare necesare pentru a materializa `spec.windows` ferestre complete.

    Prima fereastră consumă `train_bars + test_bars`; fiecare fereastră ulterioară avansează
    originea cu `step_bars`. Fereastra `k` (0-indexată) se termină la
    `k * step_bars + train_bars + test_bars`.
    """
    last = spec.windows - 1
    return last * spec.step_bars + spec.train_bars + spec.test_bars


def build_walk_forward_windows(
    spec: WalkForwardSpec, total_bars: int
) -> tuple[WalkForwardWindow, ...]:
    """Materializează ferestrele walk-forward din specificația fixată și numărul de bare.

    `spec` (din pre-înregistrare) fixează lungimea, pasul și recalibrarea (Req 19.2); aici doar
    aplicăm aceste reguli deterministe. `spec.windows >= 5` este deja impus de `WalkForwardSpec`,
    deci rezultatul respectă Req 19.1. Dacă `total_bars` nu acoperă toate ferestrele cerute,
    ridicăm `InsufficientDataError` în loc să reducem tăcut numărul de ferestre (fail-closed).

    Returnează un tuplu imuabil de `WalkForwardWindow`, în ordine cronologică.
    """
    if total_bars < 0:
        raise ValueError("total_bars nu poate fi negativ")
    needed = min_bars_required(spec)
    if total_bars < needed:
        raise InsufficientDataError(
            f"sunt necesare {needed} bare pentru {spec.windows} ferestre, "
            f"dar sunt doar {total_bars}"
        )

    windows: list[WalkForwardWindow] = []
    for k in range(spec.windows):
        dev_start = k * spec.step_bars
        dev_end = dev_start + spec.train_bars
        oos_start = dev_end
        oos_end = oos_start + spec.test_bars
        windows.append(
            WalkForwardWindow(
                index=k,
                dev_start=dev_start,
                dev_end=dev_end,
                oos_start=oos_start,
                oos_end=oos_end,
                recalibrate=spec.recalibrate,
            )
        )
    return tuple(windows)
