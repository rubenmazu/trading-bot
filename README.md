# Quant_Trading_System (`qts`)

Sistem determinist de cercetare și trading algoritmic, bazat exclusiv pe reguli matematice sau
statistice versionate și explicabile. Orizont 5–60 de minute (nu HFT).

> **Etapa inițială — fără ordine reale.** Codul care poate trimite ordine pe bani reali (Live) nu
> este inclus. Sistemul rulează doar în Backtest, Shadow și Demo (cont paper, bani virtuali).
> Avertisment: trading-ul implică risc; profitul nu este garantat, pierderile sunt posibile.

## Cerințe

- **Python 3.13** (exact; vezi `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — managerul de pachete și mediu
- **Windows** pentru fluxurile cu date live Alpaca (pachetul rulează cross-platform; adaptorul
  Alpaca folosește `alpaca-py`, instalat opțional)

## Instalare pe o mașină nouă

```bash
git clone https://github.com/rubenmazu/trading-bot.git
cd trading-bot
uv sync
```

`uv sync` instalează dependențele fixate exact din `uv.lock` (build reproductibil). Pentru fluxurile
Alpaca (Shadow/Demo pe date reale), instalează și extra-ul opțional:

```bash
uv sync --extra alpaca
```

> Mediul virtual (`.venv/`), cache-urile (`__pycache__`, `.mypy_cache`, `.pytest_cache`,
> `.ruff_cache`, `.hypothesis`) și datele locale (`data/`, `runs/`, `*.db`) **nu** sunt în git.
> Se regenerează automat pe mașina nouă; nu trebuie copiate.

## Verificare rapidă (teste, lint, tipuri)

```bash
uv run pytest            # suita completă (profil hypothesis implicit: ci)
uv run ruff check .      # lint
uv run mypy --strict src tests   # verificare de tipuri
```

Profiluri hypothesis pentru testele property-based (vezi `tests/conftest.py`):

```bash
uv run pytest --hypothesis-profile fast   # 25 de exemple (rapid, dezvoltare)
uv run pytest --hypothesis-profile ci     # 50 de exemple (implicit)
uv run pytest --hypothesis-profile deep   # 5000 de exemple (înainte de promovare)
```

## Comenzi CLI

Entry point: `qts` (definit în `pyproject.toml`). Rulează cu `uv run qts <comandă>`.

| Comandă | Ce face |
|---|---|
| `qts version` | Afișează versiunea Trading_Engine |
| `qts backtest` | Backtest pe date istorice (SimBroker + SimClock), fără conexiune reală |
| `qts shadow` | Feed curent cu execuție simulată local (SimBroker), ceas real; `--source alpaca` pentru date reale |
| `qts demo` | Cont Alpaca **paper** (bani virtuali, API real), streaming continuu; `--once` pentru o singură trecere |
| `qts evaluate` | Raportul de evaluare al cercetării (brut/costuri/net/incertitudine) |

### Backtest (fără chei, zero risc)

Generează un set de date sintetic și rulează:

```bash
uv run python scripts/gen_synthetic.py          # creează data/synthetic.csv
uv run qts backtest --config config/backtest.toml --stage-lock stage.lock
```

> Backtest-ul creează un `Configuration_Snapshot` din starea git (reproductibilitate). Repo-ul
> trebuie să aibă cel puțin un commit, altfel pornirea este refuzată.

### Shadow pe date reale Alpaca (feed real, execuție simulată local, zero risc)

```bash
uv run qts shadow --source alpaca
```

### Demo pe cont Alpaca paper (bani virtuali, API real)

```bash
uv run qts demo --once     # o singură trecere, de verificare
uv run qts demo            # continuu (comportament ca pe real), oprești cu Ctrl+C
```

## Secrete (chei API Alpaca)

Cheile **nu** sunt niciodată în cod sau în fișiere de configurație — doar ca referințe
(`secret_ref`). Valorile reale stau în Secret_Store (keyring-ul sistemului). Pe fiecare mașină unde
rulezi Shadow/Demo pe date reale, pui cheile o singură dată:

```bash
# cont paper Alpaca → API Key ID + Secret Key (din dashboard, modul Paper Trading)
uv run python -c "import keyring; keyring.set_password('qts','qts/demo/alpaca_key','API_KEY_ID')"
uv run python -c "import keyring; keyring.set_password('qts','qts/demo/alpaca_key_secret','SECRET_KEY')"
```

Pentru Shadow, referințele sunt `qts/shadow/alpaca_key` și `qts/shadow/alpaca_secret`.

> Fără chei în keyring, comenzile Shadow/Demo pe date reale **refuză pornirea** cu un mesaj clar
> (cod de ieșire 2), fără a trimite nimic. Nu se poate strica nimic: contul este paper.

## Orele de piață

Alpaca livrează date de piață US. Piața US este deschisă aproximativ **16:30–23:00 ora României**
(09:30–16:00 ET). În afara orelor, `qts demo --once` arată ultimele bare disponibile și se oprește;
modul continuu așteaptă bare noi care apar doar la deschiderea pieței.

## Structura proiectului

```
config/            # TOML per mediu (backtest, shadow, demo); live refuzat
src/qts/
  core/            # modele, bani (Decimal), ceas, bus, motor event-driven
  data/            # ingestie, validare OHLCV, bare, prospețime, surse (CSV, Alpaca)
  strategy/        # strategii pure, deterministe (mean-reversion de referință)
  risk/            # motor de risc independent, limite, dimensionare, eligibilitate
  oms/             # ciclul de viață al ordinelor (FSM), idempotență
  portfolio/       # poziții, numerar, PnL (brut vs. net)
  costs/           # Complete_Cost_Model (spread, comision, slippage, latență, FX, taxe)
  broker/          # adaptoare: SimBroker, FakeBroker, Alpaca (paper), fail-safe, fabrică
  recon/           # reconciliere broker ↔ intern (fail-closed)
  safety/          # kill switch, live gate (închis), etapă, decizii deschise
  persistence/     # SQLite (WAL), jurnal append-only cu lanț SHA-256, audit, recuperare
  secrets/         # Secret_Store (keyring), redactarea secretelor
  health/          # monitor de sănătate, alerte, metrici
  research/        # pre-înregistrare, partiții OOS, walk-forward, bootstrap, raport
  cli.py  bootstrap.py
tests/             # unit, property (hypothesis), integration, contract, fixtures
scripts/           # unelte (ex. generator de date sintetice)
stage.lock         # etapa proiectului ("initial"); blochează Live
```

## Siguranță prin construcție

- **Fără ordine reale în etapa inițială**: `stage.lock = "initial"` + un `Fail_Safe_Block`
  independent resping orice `Real_Order_Request` înainte de a ajunge la broker.
- **Demo = Live, doar cu bani virtuali**: același motor în toate modurile; diferă numai
  adaptoarele, configurația de mediu și referințele credențialelor.
- **Fail-closed**: orice dată lipsă, expirată sau neconcordantă blochează ordinele noi.
- **Limite de risc**: capital de referință 100 EUR, pierdere totală maximă 10 EUR, 0,25–0,50 EUR
  per tranzacție; kill switch permanent la atingerea limitei totale.
- **Audit**: fiecare eveniment este jurnalizat cu lanț SHA-256; starea e reconstruibilă din jurnal.

## Licență și notă

Acest proiect este personal și nu reprezintă consultanță financiară. Nu promite performanță
viitoare. Folosește-l pe propriul risc.
