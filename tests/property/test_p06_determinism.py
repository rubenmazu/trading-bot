"""P6: determinism byte cu byte pentru același artefact, aceleași date și aceeași sămânță.

Aceeași strategie, cu aceleași date, parametri și sămânță produce secvențe identice (byte cu
byte, după serializare canonică) de semnale, ordine, execuții și metrici (Req 6.1, 7.6, 17.5).

Montaj. Pentru fiecare exemplu generăm un set de date sintetic (scenariu / sămânță de date /
`n_bars` / volatilitate) și parametri de strategie (`lookback`, `entry_z`, `exit_z`, `stop_k`)
plus o sămânță `run.seed`. Rulăm `run_backtest` de DOUĂ ori cu intrări identice, dar în două
directoare de lucru diferite (`base_dir` diferit), deci căile fizice ale bazei de date și ale
CSV-ului diferă.

Egalitatea `snapshot_id` (determinismul artefactului). `snapshot_id` hash-uiește configurația
efectivă, care include `run.db_path` și `data.dataset_path` ca TEXT, plus `dataset_id`.
`dataset_id` este derivat din suma de control a conținutului CSV (nu din cale), iar
`created_at` al snapshot-ului este `manifest.start` (timp simulat, nu ceasul de perete). De
aceea păstrăm aceleași șiruri RELATIVE pentru `dataset_path` și `db_path` în ambele configurații
și scriem conținut CSV identic; numai `base_dir` (locația de lucru) diferă, deci căile se
rezolvă în fișiere fizice diferite, dar configurația efectivă este identică → `snapshot_id`
coincide. `code_version` este injectat fix (un depozit temporar nu are HEAD git).

Determinismul fluxului. Comparăm `BacktestResult` câmp cu câmp (ordine, execuții, brut realizat
și nerealizat, costuri pe categorii, net, stările ordinelor, evenimente procesate, `snapshot_id`
și `journal_head` – `seq` ȘI hash-ul de cap, adică întregul lanț byte cu byte). Apoi citim ambele
jurnale integral și comparăm fiecare `Audit_Record` canonicalizat (inclusiv `hash` și
`prev_hash`): dacă fluxurile sunt identice byte cu byte, listele canonice sunt egale. Nu există
niciun câmp care să difere legitim: timpii provin din `SimClock` avansat de evenimente, nu din
ceasul de perete.

Non-vacuitate și sanity. Verificăm că jurnalul conține semnale (strategia rulează) și cerem ca
cel puțin unele exemple să producă ordine și execuții. Separat, verificăm că o sămânță DIFERITĂ
(fie a datelor, fie `run.seed`) schimbă în general ieșirea: cel puțin setul de date diferă, deci
`dataset_id` ori un câmp de rezultat diferă (verificare laxă, pentru a evita coliziunile rare).
"""

from __future__ import annotations

import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from qts.bootstrap import BacktestResult, run_backtest
from qts.config.snapshot import CodeVersion
from qts.core.models import AuditRecord, canonical_bytes, canonical_json
from qts.persistence.db import open_db
from qts.persistence.journal import Journal, hash_fields
from tests.fixtures.synthetic import SCENARIOS, SyntheticSpec, write_synthetic_dataset

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "config"
STAGE_LOCK = REPO / "stage.lock"
CODE = CodeVersion(git_commit="0" * 40, git_dirty=False, lock_sha256="d" * 64)

# Căi RELATIVE identice în ambele rulări: numai `base_dir` diferă, deci configurația efectivă
# (și astfel `snapshot_id`) rămâne identică, dar fișierele fizice sunt separate.
DATASET_REL = "data/synthetic.csv"
DB_REL = "run.db"


# ----------------------------------------------------------------------------- generatoare


@st.composite
def _spec(draw: st.DrawFn) -> SyntheticSpec:
    """Set de date sintetic valid, modest ca mărime pentru viteză."""
    return SyntheticSpec(
        scenario=draw(st.sampled_from(SCENARIOS)),
        seed=draw(st.integers(0, 2**32 - 1)),
        n_bars=draw(st.integers(30, 120)),
        volatility=draw(st.sampled_from((0.002, 0.004, 0.008, 0.02))),
    )


@st.composite
def _params(draw: st.DrawFn) -> dict[str, str]:
    """Parametri de strategie în intervale valide (vezi MeanReversionParams)."""
    entry = draw(st.sampled_from(("0.5", "1.0", "1.5", "2.0", "2.5")))
    exit_z = draw(st.sampled_from(("-0.25", "0", "0.25", "0.5")))
    stop_k = draw(st.sampled_from(("1.0", "2.0", "3.0", "4.0")))
    lookback = draw(st.integers(2, 30))
    return {
        "lookback": str(lookback),
        "entry_z": entry,
        "exit_z": exit_z,
        "stop_k": stop_k,
    }


def _params_toml(params: dict[str, str]) -> str:
    """Secțiunea `[strategy.params]` în TOML (lookback întreg, restul ca text)."""
    lines = ["", "[strategy.params]", f"lookback = {params['lookback']}"]
    for key in ("entry_z", "exit_z", "stop_k"):
        lines.append(f'{key} = "{params[key]}"')
    return "\n".join(lines) + "\n"


def _write_run_dir(
    base_dir: Path, spec: SyntheticSpec, params: dict[str, str], run_seed: int
) -> Path:
    """Scrie setul sintetic și o configurație de backtest cu căi RELATIVE identice."""
    data_dir = base_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    # Nume de fișier fix: `data_file` din manifest este doar numele, deci nu intră în căi.
    write_synthetic_dataset(data_dir, spec, filename="synthetic.csv")

    text = (CONFIG_DIR / "backtest.toml").read_text(encoding="utf-8")
    edits = {
        'dataset_path = "data/synthetic.csv"': f'dataset_path = "{DATASET_REL}"',
        'db_path = "runs/backtest.db"': f'db_path = "{DB_REL}"',
        "seed = 42": f"seed = {run_seed}",
    }
    for old, new in edits.items():
        assert old in text, old
        text = text.replace(old, new)
    text += _params_toml(params)
    config_path = base_dir / "backtest.toml"
    config_path.write_text(text, encoding="utf-8")
    return config_path


def _run(
    base_dir: Path, spec: SyntheticSpec, params: dict[str, str], run_seed: int
) -> BacktestResult:
    base_dir.mkdir(parents=True, exist_ok=True)
    config_path = _write_run_dir(base_dir, spec, params, run_seed)
    return run_backtest(
        config_path,
        stage_lock=STAGE_LOCK,
        base_dir=base_dir,
        repo_root=base_dir,
        code_version=CODE,
    )


@contextmanager
def _work_dir() -> Iterator[Path]:
    """Director de lucru nou per exemplu (hypothesis nu resetează fixture-urile `tmp_path`)."""
    with tempfile.TemporaryDirectory(prefix="p06_") as name:
        yield Path(name)


def _read_journal(db_path: str) -> list[AuditRecord]:
    conn = open_db(db_path)
    try:
        return list(Journal(conn).read())
    finally:
        conn.close()


def _result_fields(result: BacktestResult) -> dict[str, Any]:
    """Câmpurile de rezultat care trebuie să coincidă byte cu byte (fără `db_path`)."""
    return {
        "snapshot_id": result.snapshot_id,
        "run_id": result.run_id,
        "dataset_id": result.dataset_id,
        "events_processed": result.events_processed,
        "orders": result.orders,
        "fills": result.fills,
        "realized_gross_eur": result.realized_gross_eur,
        "unrealized_gross_eur": result.unrealized_gross_eur,
        "gross_eur": result.gross_eur,
        "net_eur": result.net_eur,
        "costs": canonical_json(result.costs),
        "costs_spread": result.costs.spread,
        "costs_commission": result.costs.commission,
        "costs_slippage": result.costs.slippage,
        "costs_latency": result.costs.latency,
        "costs_fx_conversion": result.costs.fx_conversion,
        "costs_taxes": result.costs.taxes,
        "costs_total": result.costs.total,
        "journal_head": result.journal_head,  # (seq, head_hash) — lanțul byte cu byte
        "journal_verified": result.journal_verified,
        "rejected_rows": result.rejected_rows,
        "order_states": tuple(sorted(result.order_states.items())),
    }


# ----------------------------------------------------------------------------- proprietatea


@settings(max_examples=40)
@given(spec=_spec(), params=_params(), run_seed=st.integers(0, 2**32 - 1))
def test_property_6_backtest_is_byte_for_byte_deterministic(
    spec: SyntheticSpec, params: dict[str, str], run_seed: int
) -> None:
    """**Validates: Requirements 6.1, 7.6, 17.5**"""
    with _work_dir() as root:
        a = _run(root / "a", spec, params, run_seed)
        b = _run(root / "b", spec, params, run_seed)

        # Căile fizice diferă (artefacte separate), dar artefactul logic este același.
        assert a.db_path != b.db_path

        # 1) BacktestResult identic pe toate câmpurile relevante, inclusiv snapshot_id și lanțul.
        fa, fb = _result_fields(a), _result_fields(b)
        assert fa == fb, f"BacktestResult diferă: {fa} != {fb}"

        # 2) Fluxurile de jurnal identice byte cu byte după canonicalizare (hash/prev_hash incluse).
        records_a = _read_journal(a.db_path)
        records_b = _read_journal(b.db_path)
        assert len(records_a) == len(records_b)

        canon_a = [canonical_bytes(r) for r in records_a]
        canon_b = [canonical_bytes(r) for r in records_b]
        for i, (ra, rb, ba, bb) in enumerate(
            zip(records_a, records_b, canon_a, canon_b, strict=True)
        ):
            # Câmpurile folosite la hash sunt identice (seq, ts, type, payload, prev_hash, ...).
            assert hash_fields(ra) == hash_fields(rb), (
                f"primul Audit_Record diferit la index {i} (seq={ra.seq}): "
                f"{hash_fields(ra)} != {hash_fields(rb)}"
            )
            # Lanțul în sine: hash și prev_hash coincid, deci octeții sunt identici.
            assert ba == bb, f"Audit_Record diferit byte cu byte la index {i} (seq={ra.seq})"

        assert canon_a == canon_b

        # 3) Non-vacuitate: strategia a rulat (există semnale în jurnal).
        assert any(r.type == "signal" for r in records_a)


@settings(max_examples=40)
@given(spec=_spec(), params=_params(), run_seed=st.integers(0, 2**32 - 1))
def test_property_6_different_seed_changes_output(
    spec: SyntheticSpec, params: dict[str, str], run_seed: int
) -> None:
    """Sanity (non-vacuitate): o sămânță diferită schimbă în general ieșirea.

    **Validates: Requirements 6.1, 7.6, 17.5**
    """
    other_data_seed = (spec.seed + 1) % (2**32)
    other_spec = SyntheticSpec(
        scenario=spec.scenario,
        seed=other_data_seed,
        n_bars=spec.n_bars,
        volatility=spec.volatility,
    )

    with _work_dir() as root:
        base = _run(root / "base", spec, params, run_seed)
        changed = _run(root / "changed", other_spec, params, run_seed)

    # Setul de date diferă prin sămânță → `dataset_id` diferă aproape sigur; verificare laxă
    # pentru a tolera coliziunile rare de conținut.
    differs = (
        base.dataset_id != changed.dataset_id
        or base.snapshot_id != changed.snapshot_id
        or _result_fields(base) != _result_fields(changed)
    )
    assert differs, "o sămânță de date diferită nu a schimbat nimic (coliziune improbabilă)"


# Non-vacuitate la nivel de suită: cel puțin un montaj produce ordine și execuții reale.


def test_property_6_nonvacuous_produces_orders_and_fills(tmp_path: Path) -> None:
    """Montaj determinist care emite ordine și execuții (event()), pentru a nu rula în gol."""
    spec = SyntheticSpec(scenario="mean_reverting", seed=11, n_bars=120, volatility=0.004)
    params = {"lookback": "20", "entry_z": "1.0", "exit_z": "0", "stop_k": "3.0"}
    result = _run(tmp_path / "nv", spec, params, run_seed=7)
    assert result.orders > 0 and result.fills > 0
    assert result.journal_verified
