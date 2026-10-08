# Implementation Plan

## Overview

Fiecare task se construiește pe cele anterioare și se termină cu cod integrat și testat. Proprietățile de corectitudine (P1–P16) sunt definite în `design.md`. Pe toată durata acestui plan nu se scrie niciun adaptor live și nu se trimite niciun ordin real.

## Tasks

- [x] 1. Schelet proiect și unelte de calitate
  - Creează `pyproject.toml` (Python 3.13, `uv`, dependențe fixate exact: pydantic, numpy, polars, keyring, typer, pytest, hypothesis, ruff, mypy), `uv.lock`, structura `src/qts/` și `tests/` din design
  - Configurează `ruff`, `mypy --strict`, profilurile hypothesis `ci` și `deep`, `.gitignore` (exclude `*.db`, `data/`, `.env`)
  - Creează `stage.lock` cu `project_stage = "initial"`
  - Adaugă un test fum care importă pachetul; `uv run pytest`, `ruff`, `mypy` trec
  - _Requirements: 17.1, 30.7_

- [x] 2. Nucleu: bani, timp, modele de domeniu
- [x] 2.1 Implementează `core/money.py` (context `Decimal`, rotunjire la tick/pas, conversii) și `core/clock.py` (`SimClock`, `WallClock`, UTC obligatoriu)
  - _Requirements: 6.1, 17.5_
- [x] 2.2 Implementează modelele pydantic imuabile din design (`Instrument`, `Bar`, `MarketEvent`, `Signal`, `OrderIntent`, `Order`, `ExecutionEvent`, `CostBreakdown`, `AuditRecord`) cu serializare canonică
  - _Requirements: 5.2, 6.3, 6.5, 24.3_
- [x] 2.3 Teste unitare pentru rotunjire și validarea modelelor
  - _Requirements: 13.9_

- [x] 3. Configurație, etapă și snapshot
- [x] 3.1 Implementează `config/schema.py` și `config/loader.py`: TOML per mediu, `extra="forbid"`, raportarea tuturor abaterilor, identificator explicit de mediu
  - _Requirements: 1.3, 17.1, 17.2, 17.6_
- [x] 3.2 Implementează `safety/stage.py`: citește `stage.lock`, refuză pornirea când `Initial_Stage` are mod Live sau endpoint/cont live dintr-o listă versionată
  - _Requirements: 2.2, 2.3, 2.4_
- [x] 3.3 Implementează `config/snapshot.py`: Configuration_Snapshot hash-uit (git HEAD, hash `uv.lock`, config efectivă fără secrete, sămânță); oprire dacă snapshot-ul nu poate fi creat
  - _Requirements: 1.4, 17.3, 17.4, 17.7_
- [x] 3.4 Teste unitare pentru configurații invalide și refuzul live în etapa inițială
  - _Requirements: 2.4, 17.2_

- [x] 4. Persistență, jurnal și audit
- [x] 4.1 Implementează `persistence/db.py` (SQLite WAL, `synchronous=FULL`, migrații versionate) și `persistence/journal.py` (append-only, `seq` monoton, lanț SHA-256)
  - _Requirements: 24.1, 24.3, 24.4, 26.1_
- [x] 4.2 Implementează `persistence/audit.py`: verificarea lanțului, export cu versiunea schemei și dovada integrității, reconstrucția lanțului Market_Event–Signal–Order_Intent–risc–Execution_Event
  - _Requirements: 24.5, 24.7_
- [x] 4.3 Implementează `secrets/store.py` (`keyring`, `SecretValue` mascat) și un filtru de redactare pentru loguri, jurnal și erori
  - _Requirements: 23.1, 23.2, 23.4, 23.7_
- [x] 4.4 Test de proprietate P12: orice modificare, ștergere sau reordonare în jurnal este detectată
  - _Requirements: 24.4_
- [x] 4.5 Test de proprietate P15: secretele generate nu apar în nicio ieșire persistentă
  - _Requirements: 23.2_

- [x] 5. Date de piață
- [x] 5.1 Implementează `data/validate.py` (invariante OHLCV, ordine temporală, câmpuri obligatorii) și `data/normalize.py` (schemă unică, deduplicare pe cheie canonică)
  - _Requirements: 5.3, 5.4, 5.7_
- [x] 5.2 Implementează `data/manifest.py` și `data/csv_source.py`: încărcare CSV/Parquet cu manifest, verificarea sumei de control, invalidarea setului corupt
  - _Requirements: 5.1, 5.6, 5.8_
- [x] 5.3 Implementează `data/bars.py` (agregare 5–60 min, refuz în afara intervalului) și `data/freshness.py` (blocare pe instrument la date expirate)
  - _Requirements: 4.5, 5.5_
- [x] 5.4 Generator de date sintetice deterministe (random walk, trend, mean-reverting, goluri) în `tests/fixtures/`
  - _Requirements: 7.4, 7.7_
- [x] 5.5 Teste unitare pentru bare invalide, duplicate, sume de control și prospețime
  - _Requirements: 5.4, 5.5, 5.7, 5.8_

- [x] 6. Model de costuri
- [x] 6.1 Implementează `costs/model.py`: spread, comision (tabele versionate, minim/maxim), slippage, latență, conversie FX, taxe configurate; `CostModelIncomplete` când lipsește o componentă
  - _Requirements: 8.1, 8.2, 8.3_
- [x] 6.2 Suportă multiplicatori de stres (×1,5, ×2,0) și latență de stres
  - _Requirements: 20.3, 20.4_
- [x] 6.3 Test de proprietate P14: rezultatul net nu depășește rezultatul brut și este monoton descrescător în multiplicatorul costurilor
  - _Requirements: 8.4, 20.3_

- [x] 7. Strategie
- [x] 7.1 Implementează `strategy/base.py` (protocol pur) și `strategy/history_view.py`, care ridică excepție la orice acces la date viitoare
  - _Requirements: 6.1, 6.2, 6.6, 7.2_
- [x] 7.2 Implementează `strategy/mean_reversion.py` (z-score pe 15 min, ieșire la medie sau la stop, `NONE` cu motiv la date invalide, inputurile și regulile incluse în `Signal`)
  - _Requirements: 6.3, 6.4_
- [x] 7.3 Test de proprietate P7: modificarea barelor de după *t* nu schimbă semnalele emise până la *t*
  - _Requirements: 7.2_

- [x] 8. Motor de risc
- [x] 8.1 Implementează `risk/context.py`, `risk/sizing.py`, `risk/limits.py` și `risk/engine.py` cu verificările ordonate din design: kill switch, interdicții (levier, short, CFD, futures), dimensionare pe stop și costuri, limitele per tranzacție, zilnică și totală, reevaluare după reducere
  - _Requirements: 12.1, 12.3, 12.4, 12.6, 12.7, 13.1–13.10_
- [x] 8.2 Implementează monitorizarea continuă a pierderii zilnice și totale, cu activarea `Kill_Switch(DAY)` și `Kill_Switch(CAPITAL_CONFIG, permanent)`
  - _Requirements: 13.4, 13.6, 13.11_
- [x] 8.3 Implementează eligibilitatea instrumentelor (praguri versionate, neeligibil când bugetul nu acoperă costurile minime)
  - _Requirements: 4.1–4.4, 4.6, 4.7_
- [x] 8.4 Test de proprietate P2: orice ordin aprobat respectă limita de 0,50 EUR și numerarul disponibil
  - _Requirements: 13.2, 13.8, 13.9_
- [x] 8.5 Test de proprietate P3: după atingerea limitei totale nu se mai aprobă niciun ordin în aceeași configurație de capital
  - _Requirements: 13.5, 13.6, 13.11_
- [x] 8.6 Test de proprietate P4: după atingerea limitei zilnice nu se mai aprobă ordine în aceeași zi
  - _Requirements: 13.3, 13.4_
- [x] 8.7 Test de proprietate P5: un ordin redus respectă toate limitele
  - _Requirements: 12.7_

- [x] 9. Checkpoint: toate testele, `ruff` și `mypy` trec; operatorul răspunde la întrebările deschise înainte de a continua

- [x] 10. Gestionarea ordinelor și portofoliul
- [x] 10.1 Implementează `oms/fsm.py`: tabel static de tranziții, inclusiv stările `UNKNOWN` și `CANCEL_PENDING`; tranzițiile invalide sunt respinse fără modificarea stării
  - _Requirements: 9.1, 9.2, 9.4_
- [x] 10.2 Implementează `oms/idempotency.py` (`client_order_id` determinist, deduplicare `broker_exec_id`) și `oms/manager.py` (execuții parțiale, anulări, înghețarea instrumentului la `UNKNOWN`, cereri pentru golurile de secvență)
  - _Requirements: 9.3, 9.5, 9.6, 10.1–10.5_
- [x] 10.3 Implementează `portfolio/portfolio.py` și `portfolio/pnl.py`: poziții, numerar, PnL realizat și nerealizat, brut vs. net pe categorii de cost
  - _Requirements: 8.4, 29.1_
- [x] 10.4 Test de proprietate P8: duplicatele și reordonările execuțiilor nu schimbă starea finală
  - _Requirements: 10.2, 10.3_
- [x] 10.5 Teste de proprietate P9 și P10: conservarea cantității și FSM închis
  - _Requirements: 9.2, 9.3, 9.4_

- [x] 11. Broker simulat, fail-safe și kill switch
- [x] 11.1 Implementează `broker/adapter.py` (protocol, capabilități, cod de motiv pentru capabilități lipsă) și `broker/sim.py` (execuție la bara *t+1* cu modelul de costuri și latență, execuții parțiale configurabile)
  - _Requirements: 3.1, 3.2, 3.4, 7.3_
- [x] 11.2 Implementează `broker/fake.py` cu defecte injectabile: respingeri, timeout, duplicate, reordonări, deconectare, snapshot incomplet
  - _Requirements: 9.1, 10.4, 11.1_
- [x] 11.3 Implementează `safety/kill_switch.py` (domenii, persistență, politica `keep`/`cancel`, implicit `keep`) și `broker/fail_safe.py` (verificare sincronă a mediului, contului și porții înaintea fiecărui `submit`)
  - _Requirements: 2.1, 2.5, 2.6, 14.1–14.8, 15.9_
- [x] 11.4 Implementează `safety/live_gate.py` doar ca poartă închisă: în `Initial_Stage` evaluează condițiile și întoarce lista celor nesatisfăcute; fără cale de deschidere
  - _Requirements: 3.6, 15.1–15.5, 15.8, 15.12_
- [x] 11.5 Implementează `broker/factory.py`: refuză `environment="live"` în `Initial_Stage`, iar fiecare adaptor este învelit în `Fail_Safe_Block`
  - _Requirements: 1.2, 2.1, 2.2_
- [x] 11.6 Teste de proprietate P1 și P13: zero ordine live în etapa inițială; zero ordine noi cât timp kill switch-ul este activ
  - _Requirements: 2.1, 2.6, 14.1, 14.2_
- [x] 11.7 Suită contract comună pentru adaptoarele broker, rulată pe `SimBroker` și `FakeBroker`
  - _Requirements: 3.1–3.4_

- [x] 12. Motorul event-driven și backtestul
- [x] 12.1 Implementează `core/bus.py` (coadă prioritizată `(ts, prioritate_tip, seq)`) și `core/engine.py` (un singur fir, aplicare write-ahead în jurnal, aceeași succesiune de pași în toate modurile)
  - _Requirements: 1.1, 7.1, 12.1_
- [x] 12.2 Implementează `bootstrap.py`, care compune porturile per mod (Backtest: replay istoric + `SimBroker` + `SimClock`), și comanda CLI `qts backtest`
  - _Requirements: 1.2, 1.5, 16.6_
- [x] 12.3 Implementează invalidarea subperioadelor cu goluri peste prag și componența istorică a universului, când există
  - _Requirements: 7.4, 7.5, 7.7_
- [x] 12.4 Test de integrare: backtest pe date sintetice cu rezultat calculat manual (semnale, ordine, execuții, costuri, PnL net)
  - _Requirements: 7.3, 8.4_
- [x] 12.5 Test de proprietate P6: determinism byte cu byte pentru același artefact, aceleași date și aceeași sămânță
  - _Requirements: 6.1, 7.6, 17.5_

- [x] 13. Reconciliere și recuperare
- [x] 13.1 Implementează `recon/reconciler.py`: comparare cu snapshot-ul brokerului, toleranțe versionate, blocare pe instrument sau globală, `Kill_Switch` la snapshot incomplet, reluare cu cauză, corecție și aprobare
  - _Requirements: 11.1–11.7_
- [x] 13.2 Implementează `persistence/recovery.py`: checkpoint, validarea integrității la pornire, reluare de la primul eveniment neconfirmat, menținerea blocării până la confirmarea ordinelor în așteptare
  - _Requirements: 26.1–26.5, 26.8, 26.9_
- [x] 13.3 Test de proprietate P11: reconstruirea din jurnal este echivalentă cu procesarea online pentru orice prefix
  - _Requirements: 26.1, 26.3_
- [x] 13.4 Teste de integrare cu `FakeBroker`: cădere între commit și `submit`, reconectare, execuție în timpul anulării, snapshot incomplet
  - _Requirements: 10.2, 11.1, 11.6, 26.4_

- [x] 14. Sănătate, alerte și modul Shadow
- [x] 14.1 Implementează `health/monitor.py` (stări per componentă, verificarea ceasului NTP, blocare la stare critică) și `health/alerts.py` (coadă persistentă cu reîncercare, livrare în consolă/log)
  - _Requirements: 25.1–25.6, 26.6, 26.7_
- [x] 14.2 Adaugă metrici (latențe, prospețime, ordine pe stare, costuri, PnL, utilizarea limitelor) și benchmarkul p50/p95/p99 pe `Supported_Load` inițial
  - _Requirements: 27.1–27.5_
- [x] 14.3 Compune modul Shadow (feed curent printr-un `Data_Adapter` de redare cu ceas real + `SimBroker`) și comanda `qts shadow`; adaptorul de date real se adaugă după alegerea sursei
  - _Requirements: 1.1, 16.1_

- [x] 15. Checkpoint: toate testele trec, inclusiv profilul hypothesis `deep`; operatorul răspunde la întrebările deschise

- [ ] 16. Pipeline de cercetare și validare
- [ ] 16.1 Implementează `research/preregistration.py` (salvare hash-uită; evaluarea este blocată fără pre-înregistrare) și `research/partition.py` (separare cronologică, registru OOS cu consum unic, reclasificare la contaminare)
  - _Requirements: 18.1–18.6, 19.2, 19.8, 21.1, 21.8_
- [x] 16.2 Implementează `research/walk_forward.py` (≥5 ferestre) și `research/bootstrap.py` (stationary bootstrap, ≥10.000 de reeșantionări, sămânță, distribuții parțiale)
  - _Requirements: 19.1, 19.3, 19.4, 19.7_
- [x] 16.3 Implementează `research/stress.py`, `research/regimes.py`, `research/robustness.py` (vecinătăți, eliminarea celor mai mari N câștiguri) și `research/multiple_testing.py` (Deflated Sharpe, Holm–Bonferroni)
  - _Requirements: 19.5, 19.6, 20.1–20.6, 21.2–21.6_
- [x] 16.4 Implementează `research/report.py` și comanda `qts evaluate`: brut, costuri, net, ipoteze, incertitudine, toate variantele, marcare „incomplet”, avertismentul că profitul nu este garantat
  - _Requirements: 21.3, 21.7, 29.1–29.5_
- [x] 16.5 Test de proprietate P16: un OOS nu poate fi evaluat a doua oară
  - _Requirements: 18.3, 18.6_
- [x] 16.6 Test statistic: pe date random walk pipeline-ul respinge strategia în ≥95% din rulări
  - _Requirements: 8.5, 21.4, 21.5_

- [ ] 17. Artefacte, promovare și decizii deschise
- [x] 17.1 Implementează `StrategyArtifact` imuabil și promovarea strict în ordinea Backtest → Shadow → Demo, cu detectarea diferențelor din afara adaptoarelor/configurației de mediu și starea terminală `Respins`
  - _Requirements: 16.1–16.8, 22.3, 22.7_
- [x] 17.2 Implementează registrul `Open_Decision` și aprobările (opțiuni, criterii, responsabil, termen, dovezi) cu blocarea etapelor dependente; populează deciziile: broker, univers, sursă de date, durata Demo
  - _Requirements: 3.5, 22.1, 22.2, 22.8, 30.1–30.9_
- [~] 17.3 Implementează evaluarea `Paper_Qualification` față de criteriile pre-înregistrate
  - _Requirements: 22.4–22.6_
- [x] 17.4 Implementează regulile pentru modificarea capitalului (`Capital_Change_Authorization`, fără creșteri automate, limita totală ≤ 10 EUR)
  - _Requirements: 28.1–28.7_

- [ ] 18. Checkpoint final: toate testele, `ruff` și `mypy` trec; backtest end-to-end pe date sintetice prin CLI; operatorul răspunde la întrebările deschise

## Notes

- Adaptorul Demo concret, adaptorul de date real și orice cod Live nu fac parte din acest plan. Ele depind de deciziile deschise (broker, sursă de date) și de ieșirea aprobată din `Initial_Stage`.
- Testele de proprietate rulează cu profilul `ci` la fiecare build și cu profilul `deep` la checkpoint-uri.

## Task Dependency Graph

```json
{
  "waves": [
    { "id": 0, "tasks": ["1"] },
    { "id": 1, "tasks": ["2.1", "2.2"] },
    { "id": 2, "tasks": ["2.3", "3.1", "3.2", "4.1", "4.3", "5.1", "6.1"] },
    { "id": 3, "tasks": ["3.3", "3.4", "4.2", "4.4", "4.5", "5.2", "5.3", "6.2", "7.1"] },
    { "id": 4, "tasks": ["5.4", "5.5", "6.3", "7.2", "8.1"] },
    { "id": 5, "tasks": ["7.3", "8.2", "8.3", "8.4", "8.7"] },
    { "id": 6, "tasks": ["8.5", "8.6"] },
    { "id": 7, "tasks": ["9"] },
    { "id": 8, "tasks": ["10.1", "10.3"] },
    { "id": 9, "tasks": ["10.2", "10.5"] },
    { "id": 10, "tasks": ["10.4", "11.1", "11.3"] },
    { "id": 11, "tasks": ["11.2", "11.4"] },
    { "id": 12, "tasks": ["11.5", "11.7"] },
    { "id": 13, "tasks": ["11.6", "12.1"] },
    { "id": 14, "tasks": ["12.2"] },
    { "id": 15, "tasks": ["12.3", "12.4", "12.5", "13.1"] },
    { "id": 16, "tasks": ["13.2"] },
    { "id": 17, "tasks": ["13.3", "13.4", "14.1"] },
    { "id": 18, "tasks": ["14.2", "14.3"] },
    { "id": 19, "tasks": ["15"] },
    { "id": 20, "tasks": ["16.1"] },
    { "id": 21, "tasks": ["16.2", "16.3", "16.5"] },
    { "id": 22, "tasks": ["16.4"] },
    { "id": 23, "tasks": ["16.6", "17.1", "17.2"] },
    { "id": 24, "tasks": ["17.3", "17.4"] },
    { "id": 25, "tasks": ["18"] }
  ]
}
```
