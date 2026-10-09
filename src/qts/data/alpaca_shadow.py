"""Fabrica Shadow pentru feed-ul Alpaca: compune `AlpacaDataAdapter` din config + Secret_Store.

Modul Shadow (`run_shadow`) primește un `ShadowDataFactory` cu semnătura
`(config, clock, base_dir) -> DataAdapter`. Aici construim un `AlpacaDataAdapter` real:

- simbolul și intervalul vin din configurația Shadow (primul instrument și `data.bar_interval_min`);
- `tick_size` este cel al instrumentului configurat;
- cheile API Alpaca vin din `Secret_Store` (keyring), prin referințe de forma
  `qts/<mediu>/alpaca_key` și `qts/<mediu>/alpaca_secret` — niciodată din config sau cod (Req 23);
- ceasul injectat (`WallClock` în producție) stabilește fereastra de timp cerută de la Alpaca.

Fail-closed: dacă cheile lipsesc din Secret_Store sau identitatea nu este autorizată, construcția
ridică o eroare clară *înaintea* procesării vreunui eveniment. Nicio comandă reală nu părăsește
procesul: Shadow folosește `SimBroker` (execuție locală).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Final

from qts.config.schema import AppConfig
from qts.core.clock import Clock
from qts.data.adapter import DataAdapter
from qts.data.alpaca_source import (
    AlpacaClientFactory,
    AlpacaCredentials,
    AlpacaDataAdapter,
    AlpacaStreamingSource,
    SleepFn,
    _default_client_factory,
    wall_clock_sleep,
)
from qts.secrets.store import Identity, SecretRef, SecretStore

__all__ = [
    "SHADOW_IDENTITY",
    "ShadowDataFactory",
    "alpaca_shadow_factory",
    "alpaca_streaming_factory",
]

# Semnătura seam-ului de date Shadow (identică cu `bootstrap.ShadowDataFactory`). Redefinită aici
# ca alias local pentru a evita un ciclu de import (bootstrap importă din qts.data, nu invers).
ShadowDataFactory = Callable[[AppConfig, Clock, Path], DataAdapter]

# Identitatea cu care rulează feed-ul Shadow la cererea secretelor (ACL în Secret_Store).
SHADOW_IDENTITY: Final = Identity("shadow-feed")

# Fereastra de istoric recent cerută la pornire (bare recente pentru a avea context de strategie).
_DEFAULT_LOOKBACK: Final = timedelta(days=5)


def alpaca_shadow_factory(
    store: SecretStore,
    *,
    lookback: timedelta = _DEFAULT_LOOKBACK,
    client_factory: AlpacaClientFactory | None = None,
) -> ShadowDataFactory:
    """Întoarce un `ShadowDataFactory` care construiește `AlpacaDataAdapter` din config + secrete.

    `store` furnizează cheile API; `client_factory` este injectabil pentru teste (implicit importă
    `alpaca-py` la rulare). Referințele cheilor sunt derivate din mediul configurației:
    `qts/<environment>/alpaca_key` și `qts/<environment>/alpaca_secret`.
    """

    def factory(config: AppConfig, clock: Clock, _base_dir: Path) -> DataAdapter:
        instrument = config.instruments[0]
        env = config.environment
        credentials = AlpacaCredentials(
            api_key_ref=SecretRef(f"qts/{env}/alpaca_key"),
            api_secret_ref=SecretRef(f"qts/{env}/alpaca_secret"),
        )
        return AlpacaDataAdapter.from_secret_store(
            symbol=instrument.symbol,
            interval_min=config.data.bar_interval_min,
            tick_size=instrument.tick_size,
            store=store,
            credentials=credentials,
            requester=SHADOW_IDENTITY,
            environment=env,
            clock_now=clock.now(),
            lookback=lookback,
            client_factory=client_factory,
        )

    return factory


def _resolve_client(
    store: SecretStore,
    env: str,
    client_factory: AlpacaClientFactory | None,
) -> object:
    """Dezvăluie cheile feed-ului din Secret_Store și construiește clientul concret (fail-closed).

    Cheile vin din `qts/<env>/alpaca_key` și `qts/<env>/alpaca_secret` (niciodată din config/cod,
    Req 23); `client_factory` este injectabil pentru teste (implicit importă `alpaca-py`).
    """
    api_key = store.get(SecretRef(f"qts/{env}/alpaca_key"), SHADOW_IDENTITY, env)
    api_secret = store.get(SecretRef(f"qts/{env}/alpaca_secret"), SHADOW_IDENTITY, env)
    factory = client_factory if client_factory is not None else _default_client_factory
    return factory(api_key.reveal(), api_secret.reveal())


def alpaca_streaming_factory(
    store: SecretStore,
    *,
    lookback: timedelta = _DEFAULT_LOOKBACK,
    client_factory: AlpacaClientFactory | None = None,
    sleep: SleepFn = wall_clock_sleep,
    max_polls: int | None = None,
    warmup: bool = True,
) -> ShadowDataFactory:
    """Întoarce o fabrică de date care construiește un `AlpacaStreamingSource` continuu.

    Aceeași semnătură ca `alpaca_shadow_factory` (`(config, clock, base_dir) -> DataAdapter`), dar
    sursa rămâne activă: emite un lot de încălzire (`lookback`), apoi barele nou închise la fiecare
    sondaj, până când seam-ul de oprire (`max_polls`) o cere. Timpul vine din ceasul injectat
    (`clock.now`) și `sleep` este injectabil — în producție `WallClock` + `wall_clock_sleep`
    (singurul somn real), în teste un ceas fals + un `sleep` determinist, fără rețea și fără
    `time.sleep`. Această sursă este neutră față de mod: Demo o folosește azi, Live o va reutiliza.
    """

    def factory(config: AppConfig, clock: Clock, _base_dir: Path) -> DataAdapter:
        instrument = config.instruments[0]
        env = config.environment
        client = _resolve_client(store, env, client_factory)
        return AlpacaStreamingSource(
            symbol=instrument.symbol,
            interval_min=config.data.bar_interval_min,
            tick_size=instrument.tick_size,
            client=client,  # type: ignore[arg-type]
            now=clock.now,
            sleep=sleep,
            lookback=lookback,
            warmup=warmup,
            max_polls=max_polls,
        )

    return factory
