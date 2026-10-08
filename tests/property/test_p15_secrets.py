"""P15: valorile secrete nu apar în loguri, jurnal, snapshot sau erori serializate (Req 23.2)."""

import io
import logging
import pickle
from datetime import UTC, datetime

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from qts.config.loader import parse_config
from qts.config.snapshot import CodeVersion, create_snapshot
from qts.persistence.db import open_db
from qts.persistence.journal import Journal
from qts.safety.stage import ProjectStage, StageInfo
from qts.secrets.store import (
    Identity,
    InMemorySecretStore,
    RedactingFilter,
    Redactor,
    SecretAccessDeniedError,
    SecretLeakError,
    SecretRef,
)
from tests.helpers import DEMO_BROKER, config_dict

secrets_st = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126), min_size=12, max_size=40
)
REF = DEMO_BROKER["secret_ref"]
WHO = Identity("broker_adapter")


def _store(value: str, redactor: Redactor) -> InMemorySecretStore:
    return InMemorySecretStore({REF: value}, {REF: {("broker_adapter", "demo")}}, redactor)


def _snapshot_json() -> str:
    """Snapshot-ul unei configurații demo al cărei broker folosește `secret_ref` = REF,
    adică exact referința prin care adaptorul obține secretul din store."""
    cfg = parse_config(config_dict(environment="demo", broker=DEMO_BROKER))
    snap = create_snapshot(
        cfg,
        StageInfo(ProjectStage.INITIAL, "t"),
        CodeVersion("a", False, "b"),
        datetime(2026, 1, 1, tzinfo=UTC),
        [],
    )
    return snap.model_dump_json()


# Conținutul legitim (non-secret) al snapshot-ului. Hypothesis extrage constante din sursă
# (de ex. "numpy.random"), iar un "secret" egal cu un fragment din acest text nu ar indica o
# scurgere, ci o coliziune a generatorului; astfel de valori sunt excluse cu `assume`.
_BASELINE_SNAPSHOT = _snapshot_json()


@given(secrets_st)
def test_property_15_secret_absent_from_outputs(secret: str) -> None:
    assume(secret not in _BASELINE_SNAPSHOT)
    redactor = Redactor()
    value = _store(secret, redactor).get(SecretRef(REF), WHO, "demo")

    # repr / str / format / excepții
    assert secret not in repr(value) and secret not in str(value) and secret not in f"{value}"
    assert secret not in str(RuntimeError(f"conectare eșuată cu {value}"))
    with pytest.raises(TypeError):
        pickle.dumps(value)

    # loguri, inclusiv argumente și traceback
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter(redactor))
    logger = logging.getLogger(f"p15.{id(redactor)}")
    logger.addHandler(handler)
    logger.propagate = False
    logger.warning("token=%s", value.reveal())
    try:
        raise ValueError(value.reveal())
    except ValueError:
        logger.exception("eroare")
    logger.removeHandler(handler)
    assert secret not in stream.getvalue()

    # jurnal: scrierea este blocată
    j = Journal(open_db(":memory:"), redactor)
    with pytest.raises(SecretLeakError):
        j.append(
            ts=datetime(2026, 1, 1, tzinfo=UTC),
            type="x",
            correlation_id="c",
            component="t",
            component_version="1",
            actor="a",
            outcome="ok",
            payload={"k": value.reveal()},
        )

    # snapshot construit după ce secretul a fost obținut prin referința din configurație:
    # conține referința, nu valoarea
    snap_json = _snapshot_json()
    assert REF in snap_json
    assert secret not in snap_json
    assert not redactor.contains_secret(snap_json)


def test_secret_access_is_scoped_to_identity_and_environment() -> None:
    store = _store("x" * 16, Redactor())
    with pytest.raises(SecretAccessDeniedError):
        store.get(SecretRef(REF), WHO, "live")
    with pytest.raises(SecretAccessDeniedError):
        store.get(SecretRef(REF), Identity("strategy"), "demo")
