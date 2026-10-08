"""Teste unitare pentru Secret_Store și redactare (Req 23.1, 23.2, 23.4, 23.7).

Nu se atinge keyring-ul real al sistemului: `KeyringSecretStore` primește un getter stub.
"""

import copy
import io
import json
import logging
import pickle

import pytest
from keyring.errors import KeyringError

from qts.secrets.store import (
    MASK,
    Identity,
    InMemorySecretStore,
    KeyringSecretStore,
    RedactingFilter,
    Redactor,
    SecretAccessDeniedError,
    SecretLeakError,
    SecretRef,
    SecretUnavailableError,
    SecretValue,
    install_log_redaction,
)

REF = "qts/demo/broker"
FAKE = "fake-test-token-0123456789"  # valoare fictivă, doar pentru teste
WHO = Identity("broker_adapter")
ACL = {REF: {("broker_adapter", "demo")}}


def test_secret_value_is_masked_and_not_serializable() -> None:
    v = SecretValue(FAKE)
    assert repr(v) == f"SecretValue({MASK})"
    assert str(v) == MASK and f"{v}" == MASK and f"{v:>40}" == MASK
    assert v.reveal() == FAKE
    with pytest.raises(TypeError):
        pickle.dumps(v)
    with pytest.raises(TypeError):
        json.dumps({"k": v})
    with pytest.raises(TypeError):
        copy.deepcopy(v)


def test_secret_value_rejects_short_values() -> None:
    with pytest.raises(ValueError, match="minimum"):
        SecretValue("short")


def test_in_memory_store_scopes_access_and_registers_value() -> None:
    redactor = Redactor()
    store = InMemorySecretStore({REF: FAKE}, ACL, redactor)
    assert store.get(SecretRef(REF), WHO, "demo").reveal() == FAKE
    assert redactor.contains_secret(f"x{FAKE}y")
    with pytest.raises(SecretAccessDeniedError):
        store.get(SecretRef(REF), WHO, "live")
    with pytest.raises(SecretAccessDeniedError):
        store.get(SecretRef(REF), Identity("strategy"), "demo")
    with pytest.raises(SecretAccessDeniedError):
        store.get(SecretRef("qts/demo/other"), WHO, "demo")


def test_denied_access_does_not_register_or_read() -> None:
    calls: list[tuple[str, str]] = []

    def getter(service: str, name: str) -> str | None:
        calls.append((service, name))
        return FAKE

    redactor = Redactor()
    store = KeyringSecretStore(ACL, redactor, get_password=getter)
    with pytest.raises(SecretAccessDeniedError):
        store.get(SecretRef(REF), WHO, "live")
    assert calls == [] and not redactor.contains_secret(FAKE)


def test_keyring_store_reads_via_backend_stub() -> None:
    redactor = Redactor()
    store = KeyringSecretStore(
        ACL, redactor, get_password=lambda s, n: FAKE if (s, n) == ("qts", REF) else None
    )
    assert store.get(SecretRef(REF), WHO, "demo").reveal() == FAKE
    assert redactor.redact(f"token={FAKE}") == f"token={MASK}"


def test_keyring_store_missing_or_failing_backend() -> None:
    missing = KeyringSecretStore(ACL, Redactor(), get_password=lambda s, n: None)
    with pytest.raises(SecretUnavailableError):
        missing.get(SecretRef(REF), WHO, "demo")

    def boom(service: str, name: str) -> str | None:
        raise KeyringError("backend down")

    failing = KeyringSecretStore(ACL, Redactor(), get_password=boom)
    with pytest.raises(SecretUnavailableError) as ei:
        failing.get(SecretRef(REF), WHO, "demo")
    assert ei.value.__cause__ is None


def test_redactor_nested_and_assert_clean() -> None:
    r = Redactor()
    r.register(FAKE)
    r.register("tiny")  # sub prag: ignorat
    assert not r.contains_secret("tiny")
    assert r.contains_secret_in({"a": [1, ("b", {FAKE: 2})]})
    with pytest.raises(SecretLeakError):
        r.assert_clean({"payload": {"token": FAKE}}, "jurnal")
    r.assert_clean({"payload": {"token": MASK}}, "jurnal")


def test_redacting_filter_masks_message_args_traceback_and_stack() -> None:
    r = Redactor()
    r.register(FAKE)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(RedactingFilter(r))
    logger = logging.getLogger("test_secrets.filter")
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.warning("token=%s", FAKE)
        logger.warning("ctx %s", FAKE, stack_info=True)
        try:
            raise RuntimeError(FAKE)
        except RuntimeError:
            logger.exception("eșec")
    finally:
        logger.removeHandler(handler)
    out = stream.getvalue()
    assert FAKE not in out
    assert f"token={MASK}" in out and "RuntimeError" in out


def test_install_log_redaction_covers_propagated_records() -> None:
    r = Redactor()
    r.register(FAKE)
    stream = io.StringIO()
    parent = logging.getLogger("test_secrets.parent")
    parent.propagate = False
    handler = logging.StreamHandler(stream)
    parent.addHandler(handler)
    try:
        install_log_redaction(parent, r)
        install_log_redaction(parent, r)  # idempotent
        assert sum(isinstance(f, RedactingFilter) for f in handler.filters) == 1
        logging.getLogger("test_secrets.parent.child").error("pw=%s", FAKE)
    finally:
        parent.removeHandler(handler)
    assert FAKE not in stream.getvalue() and MASK in stream.getvalue()
