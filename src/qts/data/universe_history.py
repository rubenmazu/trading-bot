"""Componența istorică a `Instrument_Universe` la timpul simulat (Req 7.5).

Când sursa de date oferă istoricul componenței (listări și delistări), universul folosit în
Backtest trebuie să fie cel valabil la timpul simulat, inclusiv instrumentele delistate care
erau încă tranzacționabile atunci. Nu există încă o sursă concretă (sursa de date este o
`Open_Decision`), deci acest modul este un model în memorie, bine testat, folosit *numai când
istoricul există*. Seam-ul: `bootstrap`/motorul pot primi un `UniverseHistory` opțional și, la
decizia tranzacționabilității unui instrument la un timp, îl consultă prin `active_universe`.
Fără istoric, comportamentul rămâne cel actual (universul static din configurație).

Model: fiecare instrument are un interval de valabilitate `[listed_at, delisted_at)`, ambele
UTC. `listed_at = None` înseamnă „listat dinaintea oricărui timp observabil”; `delisted_at =
None` înseamnă „încă listat”. Un instrument este activ la `ts` dacă
`listed_at <= ts < delisted_at` (listarea este inclusivă, delistarea exclusivă: în ziua
delistării instrumentul nu mai este tranzacționabil). `active_universe(ts)` întoarce setul
înghețat al simbolurilor active la `ts`. Modelul este imuabil și serializabil canonic, deci
`version` poate fi derivat din conținut pentru `StrategyArtifact.universe_version`.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Final

from pydantic import model_validator

from qts.core.clock import ensure_utc
from qts.core.models import Frozen, canonical_hash

__all__ = [
    "UNIVERSE_HISTORY_VERSION",
    "UniverseHistory",
    "UniverseListing",
]

UNIVERSE_HISTORY_VERSION: Final = "universe-history-v1"


class UniverseListing(Frozen):
    """Intervalul de valabilitate al unui instrument în univers (Req 7.5)."""

    symbol: str
    listed_at: datetime | None = None  # None = listat dinaintea oricărui timp
    delisted_at: datetime | None = None  # None = încă listat

    @model_validator(mode="after")
    def _check(self) -> UniverseListing:
        if not self.symbol:
            raise ValueError("symbol nu poate fi gol")
        listed = ensure_utc(self.listed_at) if self.listed_at is not None else None
        delisted = ensure_utc(self.delisted_at) if self.delisted_at is not None else None
        if listed is not None and delisted is not None and delisted <= listed:
            raise ValueError(
                f"{self.symbol}: delisted_at ({delisted.isoformat()}) trebuie să fie strict "
                f"după listed_at ({listed.isoformat()})"
            )
        return self

    def active_at(self, ts: datetime) -> bool:
        """True dacă instrumentul este tranzacționabil la `ts`.

        Listarea este inclusivă, delistarea exclusivă.
        """
        ts = ensure_utc(ts)
        if self.listed_at is not None and ts < ensure_utc(self.listed_at):
            return False
        return self.delisted_at is None or ts < ensure_utc(self.delisted_at)


class UniverseHistory(Frozen):
    """Istoricul componenței: pentru fiecare timp, simbolurile active (listate și nedelistate)."""

    listings: tuple[UniverseListing, ...]
    version: str = ""

    @model_validator(mode="after")
    def _unique_symbols(self) -> UniverseHistory:
        symbols = [listing.symbol for listing in self.listings]
        if len(symbols) != len(set(symbols)):
            raise ValueError("simboluri duplicate în istoricul componenței")
        return self

    @classmethod
    def build(cls, listings: Iterable[UniverseListing]) -> UniverseHistory:
        """Construiește istoricul cu `version` derivat canonic din conținut."""
        # Ordine canonică după simbol: aceeași mulțime de listări dă aceeași versiune.
        ordered = tuple(sorted(listings, key=lambda listing: listing.symbol))
        base = cls(listings=ordered, version="")
        digest = canonical_hash(base.model_copy(update={"version": ""}))
        return base.model_copy(update={"version": f"{UNIVERSE_HISTORY_VERSION}-{digest[:16]}"})

    def active_universe(self, ts: datetime) -> frozenset[str]:
        """Simbolurile active la timpul simulat `ts` (Req 7.5)."""
        ts = ensure_utc(ts)
        return frozenset(listing.symbol for listing in self.listings if listing.active_at(ts))

    def is_active(self, symbol: str, ts: datetime) -> bool:
        """True dacă `symbol` este tranzacționabil la `ts`.

        False dacă simbolul este necunoscut sau în afara intervalului de valabilitate.
        """
        for listing in self.listings:
            if listing.symbol == symbol:
                return listing.active_at(ts)
        return False
