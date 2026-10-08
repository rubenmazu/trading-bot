"""P16: un Out_Of_Sample_Set poate fi evaluat o singură dată (Req 18.3, 18.6).

Pentru orice secvență de încercări de evaluare asupra unui OOS rezervat, PRIMA consumare reușește
și marchează OOS ca „consumed”, iar ORICE consumare ulterioară (indiferent de număr sau de
`evaluation_hash`) este refuzată cu `OosAlreadyConsumedError`, starea stocată rămânând „consumed”
și necoruptă. Proprietatea se verifică și peste reconectarea bazei de date (persistență, Req 18.6).

Design: P16 — „a doua evaluare a aceluiași `Out_Of_Sample_Set` este refuzată”.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qts.persistence.db import open_db
from qts.research import (
    DataPartitioner,
    OosAlreadyConsumedError,
    OosRegistry,
    OosStatus,
    Partition,
)

# Un „ancoră” temporal fix; intervalele generate derivă din el (toate strict crescătoare).
EPOCH = datetime(2020, 1, 1, tzinfo=UTC)

# Instrumente candidate din care generatorul alege subseturi nevide, fără duplicate.
SYMBOLS = ("AAA", "BBB", "CCC", "DDD", "EEE")


@st.composite
def oos_partitions(draw: st.DrawFn) -> Partition:
    """Generează un `Out_Of_Sample_Set` valid pe un interval și instrumente arbitrare.

    Intervalul este construit din offset-uri strict crescătoare față de `EPOCH`, astfel încât
    separarea cronologică să fie mereu validă. Instrumentele sunt un subset nevid, fără duplicate.
    """
    start_days = draw(st.integers(min_value=0, max_value=2000))
    dev_days = draw(st.integers(min_value=1, max_value=2000))
    oos_days = draw(st.integers(min_value=1, max_value=2000))
    start = EPOCH + timedelta(days=start_days)
    cut = start + timedelta(days=dev_days)
    end = cut + timedelta(days=oos_days)

    count = draw(st.integers(min_value=1, max_value=len(SYMBOLS)))
    instruments = tuple(draw(st.permutations(SYMBOLS))[:count])
    label = draw(
        st.text(
            alphabet=st.characters(min_codepoint=97, max_codepoint=122),
            min_size=1,
            max_size=8,
        )
    )

    split = DataPartitioner(oos_fraction=0.3).split_at(
        start_ts=start, cut_ts=cut, end_ts=end, instruments=instruments, label=label
    )
    return split.out_of_sample


# `evaluation_hash` arbitrar, dar nevid după strip (altfel API-ul îl refuză din construcție).
eval_hashes = st.text(min_size=1, max_size=24).filter(lambda s: bool(s.strip()))


@given(
    oos=oos_partitions(),
    first_hash=eval_hashes,
    repeat_hashes=st.lists(eval_hashes, min_size=0, max_size=12),
)
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_property_16_single_consume_then_all_refused(
    oos: Partition, first_hash: str, repeat_hashes: list[str]
) -> None:
    """**Validates: Requirements 18.3, 18.6**

    Prima consumare reușește și fixează starea „consumed”; orice consumare ulterioară este
    refuzată, oricare ar fi `evaluation_hash`, iar starea stocată rămâne „consumed” și necoruptă.
    """
    reg = OosRegistry(open_db(":memory:"))
    try:
        reg.reserve(oos, decision_id="D1", preregistration_hash="pre1")
        assert reg.is_available(oos.dataset_id)

        # Prima evaluare: reușește și marchează OOS consumat.
        consumed = reg.consume(oos.dataset_id, evaluation_hash=first_hash)
        assert consumed.status is OosStatus.CONSUMED
        assert consumed.evaluation_hash == first_hash
        assert consumed.consumed_at is not None
        assert not reg.is_available(oos.dataset_id)

        # Orice încercare ulterioară (orice număr, orice hash) este refuzată, fără a corupe starea.
        for again in repeat_hashes:
            with pytest.raises(OosAlreadyConsumedError):
                reg.consume(oos.dataset_id, evaluation_hash=again)
            record = reg.get(oos.dataset_id)
            assert record is not None
            assert record.status is OosStatus.CONSUMED
            # Prima evaluare este cea reținută; refuzurile nu suprascriu nimic.
            assert record.evaluation_hash == first_hash
            assert record.consumed_at == consumed.consumed_at
            assert reg.status(oos.dataset_id) is OosStatus.CONSUMED
            assert not reg.is_available(oos.dataset_id)
    finally:
        reg.connection.close()


@given(
    oos=oos_partitions(),
    first_hash=eval_hashes,
    post_reconnect_hashes=st.lists(eval_hashes, min_size=1, max_size=8),
)
@settings(max_examples=75, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_property_16_refusal_survives_reconnect(
    oos: Partition,
    first_hash: str,
    post_reconnect_hashes: list[str],
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """**Validates: Requirements 18.3, 18.6**

    Consumul unic este durabil: după închiderea și redeschiderea bazei, OOS rămâne „consumed” și
    orice consumare ulterioară este în continuare refuzată.
    """
    path: Path = tmp_path_factory.mktemp("oos") / "research.db"

    reg = OosRegistry(open_db(path))
    reg.reserve(oos, decision_id="D1", preregistration_hash="pre1")
    first = reg.consume(oos.dataset_id, evaluation_hash=first_hash)
    assert first.status is OosStatus.CONSUMED
    reg.connection.close()

    # Reconectare: starea „consumed” persistă și al doilea consum este refuzat.
    reg2 = OosRegistry(open_db(path))
    try:
        assert reg2.status(oos.dataset_id) is OosStatus.CONSUMED
        for again in post_reconnect_hashes:
            with pytest.raises(OosAlreadyConsumedError):
                reg2.consume(oos.dataset_id, evaluation_hash=again)
            record = reg2.get(oos.dataset_id)
            assert record is not None
            assert record.status is OosStatus.CONSUMED
            assert record.evaluation_hash == first_hash
    finally:
        reg2.connection.close()
