"""Secret_Store și redactarea secretelor (Req 23).

- Configurația conține doar `SecretRef`, niciodată valori.
- `SecretValue` nu se afișează (`***`) și nu poate fi serializat cu pickle. Valoarea se obține
  doar explicit, prin `.reveal()`, în adaptor.
- Accesul este permis numai perechilor (identitate, mediu) autorizate pentru referință (23.4).
- Fiecare valoare obținută este înregistrată în `Redactor`, care o elimină din loguri și
  blochează scrierea ei în ieșiri persistente (23.2, 23.7).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, NoReturn, Protocol

MASK = "***"
MIN_SECRET_LENGTH = 8


@dataclass(frozen=True)
class SecretRef:
    ref: str


@dataclass(frozen=True)
class Identity:
    name: str


class SecretAccessDeniedError(PermissionError):
    pass


class SecretUnavailableError(RuntimeError):
    pass


class SecretLeakError(RuntimeError):
    """Incident de securitate: o valoare secretă a ajuns într-o ieșire persistentă."""


class SecretValue:
    __slots__ = ("_value",)

    def __init__(self, value: str) -> None:
        if len(value) < MIN_SECRET_LENGTH:
            raise ValueError(f"secretele trebuie să aibă minimum {MIN_SECRET_LENGTH} caractere")
        self._value = value

    def reveal(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"SecretValue({MASK})"

    def __str__(self) -> str:
        return MASK

    def __format__(self, spec: str) -> str:
        return MASK

    def __reduce__(self) -> NoReturn:
        raise TypeError("SecretValue nu poate fi serializat")


class Redactor:
    """Registru thread-safe al valorilor secrete cunoscute în proces."""

    def __init__(self) -> None:
        self._values: set[str] = set()
        self._lock = threading.Lock()

    def register(self, value: str) -> None:
        if len(value) >= MIN_SECRET_LENGTH:
            with self._lock:
                self._values.add(value)

    def _snapshot(self) -> list[str]:
        with self._lock:
            return sorted(self._values, key=len, reverse=True)

    def redact(self, text: str) -> str:
        for value in self._snapshot():
            text = text.replace(value, MASK)
        return text

    def contains_secret(self, text: str) -> bool:
        return any(value in text for value in self._snapshot())

    def contains_secret_in(self, obj: Any) -> bool:
        """Caută recursiv în chei și valori text, înaintea oricărei serializări/escapări."""
        if isinstance(obj, str):
            return self.contains_secret(obj)
        if isinstance(obj, Mapping):
            return any(
                self.contains_secret_in(k) or self.contains_secret_in(v) for k, v in obj.items()
            )
        if isinstance(obj, (list, tuple, set, frozenset)):
            return any(self.contains_secret_in(v) for v in obj)
        return self.contains_secret(str(obj))

    def assert_clean(self, obj: Any, where: str) -> None:
        if self.contains_secret_in(obj):
            raise SecretLeakError(f"valoare secretă detectată în {where}; publicare blocată")


REDACTOR = Redactor()


class RedactingFilter(logging.Filter):
    """Filtru de logging care maschează secretele în mesaj și argumente."""

    def __init__(self, redactor: Redactor = REDACTOR) -> None:
        super().__init__()
        self._redactor = redactor

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._redactor.redact(record.getMessage())
        record.args = None
        if record.exc_info:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
            record.exc_info = None
        if record.exc_text:
            record.exc_text = self._redactor.redact(record.exc_text)
        if record.stack_info:
            record.stack_info = self._redactor.redact(record.stack_info)
        return True


def install_log_redaction(
    logger: logging.Logger | None = None, redactor: Redactor = REDACTOR
) -> RedactingFilter:
    """Atașează filtrul pe toate handler-ele loggerului (implicit root).

    Filtrele de pe handler se aplică și înregistrărilor propagate din loggeri copii,
    spre deosebire de filtrele puse pe logger.
    """
    target = logger if logger is not None else logging.getLogger()
    flt = RedactingFilter(redactor)
    for handler in target.handlers:
        if not any(isinstance(f, RedactingFilter) for f in handler.filters):
            handler.addFilter(flt)
    return flt


class SecretStore(Protocol):
    def get(self, ref: SecretRef, requester: Identity, environment: str) -> SecretValue: ...


class _AclStore:
    def __init__(
        self,
        acl: Mapping[str, set[tuple[str, str]]],
        redactor: Redactor = REDACTOR,
    ) -> None:
        self._acl = {k: set(v) for k, v in acl.items()}
        self._redactor = redactor

    def _authorize(self, ref: SecretRef, requester: Identity, environment: str) -> None:
        if (requester.name, environment) not in self._acl.get(ref.ref, set()):
            raise SecretAccessDeniedError(
                f"{requester.name} nu are acces la {ref.ref} în mediul {environment}"
            )

    def _wrap(self, value: str | None, ref: SecretRef) -> SecretValue:
        if value is None:
            raise SecretUnavailableError(f"secretul {ref.ref} nu există în store")
        self._redactor.register(value)
        return SecretValue(value)


class KeyringSecretStore(_AclStore):
    """Windows Credential Manager (sau backend-ul `keyring` al sistemului)."""

    SERVICE = "qts"

    def __init__(
        self,
        acl: Mapping[str, set[tuple[str, str]]],
        redactor: Redactor = REDACTOR,
        get_password: Callable[[str, str], str | None] | None = None,
        service: str = SERVICE,
    ) -> None:
        super().__init__(acl, redactor)
        self._get_password = get_password
        self._service = service

    def get(self, ref: SecretRef, requester: Identity, environment: str) -> SecretValue:
        self._authorize(ref, requester, environment)
        from keyring.errors import KeyringError

        getter = self._get_password
        if getter is None:
            import keyring

            getter = keyring.get_password
        try:
            value: Any = getter(self._service, ref.ref)
        except KeyringError as exc:
            raise SecretUnavailableError(
                f"Secret_Store indisponibil: {type(exc).__name__}"
            ) from None
        return self._wrap(value, ref)


class InMemorySecretStore(_AclStore):
    """Implementare pentru teste."""

    def __init__(
        self,
        values: Mapping[str, str],
        acl: Mapping[str, set[tuple[str, str]]],
        redactor: Redactor = REDACTOR,
    ) -> None:
        super().__init__(acl, redactor)
        self._values = dict(values)

    def get(self, ref: SecretRef, requester: Identity, environment: str) -> SecretValue:
        self._authorize(ref, requester, environment)
        return self._wrap(self._values.get(ref.ref), ref)
