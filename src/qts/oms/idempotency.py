"""Idempotență pentru ordine și execuții (Req 10.1–10.3, 10.5).

- `idempotency_key(run_id, strategy_id, instrument, signal_seq)` produce `client_order_id`
  determinist: `SHA-256` peste forma JSON canonică a celor patru componente, trunchiat la 32 de
  caractere hex (128 de biți). Aceleași intrări produc aceeași cheie și după repornire, deci o
  retrimitere ajunge la broker cu aceeași cheie și este deduplicată acolo (10.1, 10.2).
  Lungimea fixă de 32 de caractere încape în limitele uzuale ale câmpului `client order id`.
- `IdempotencyRegistry` ține evidența cheilor de ordin deja create și a `broker_exec_id` deja
  aplicate. Pentru fiecare execuție aplicată păstrează referința la înregistrarea originală
  (`record_seq`) și amprenta conținutului, pentru ca o retransmisie să poată fi corelată cu
  operația inițială în audit (10.5) și pentru a detecta un `broker_exec_id` refolosit cu alt
  conținut. Registrul este în memorie; persistența (tabela `idempotency`) se reconstruiește din
  jurnal la recuperare și se reîncarcă prin constructor.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Final

from qts.core.models import ExecutionEvent

__all__ = [
    "KEY_LENGTH",
    "AppliedExec",
    "IdempotencyRegistry",
    "exec_fingerprint",
    "idempotency_key",
]

KEY_LENGTH: Final = 32


def idempotency_key(run_id: str, strategy_id: str, instrument: str, signal_seq: int) -> str:
    """`client_order_id = H(run_id, strategy_id, instrument, signal_seq)` (10.1)."""
    for name, value in (
        ("run_id", run_id),
        ("strategy_id", strategy_id),
        ("instrument", instrument),
    ):
        if not value.strip():
            raise ValueError(f"{name} nu poate fi gol")
    if isinstance(signal_seq, bool) or signal_seq < 0:
        raise ValueError(f"signal_seq trebuie să fie un întreg >= 0, primit {signal_seq!r}")
    body = json.dumps(
        [run_id, strategy_id, instrument, signal_seq], separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(body.encode("utf-8")).hexdigest()[:KEY_LENGTH]


def exec_fingerprint(event: ExecutionEvent) -> str:
    """Amprenta conținutului raportat de broker; exclude momentul recepției locale."""
    data = event.model_dump(mode="json", exclude={"ts_receipt"})
    body = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(body.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class AppliedExec:
    """Referința la prima aplicare a unei execuții."""

    broker_exec_id: str
    client_order_id: str
    record_seq: int  # înregistrarea OMS care a aplicat (sau a reflectat) execuția
    fingerprint: str | None  # None când execuția a fost reflectată de reconciliere


class IdempotencyRegistry:
    """Evidența cheilor de ordin și a execuțiilor aplicate."""

    def __init__(
        self,
        order_keys: Iterable[str] = (),
        applied: Iterable[AppliedExec] = (),
    ) -> None:
        self._orders: set[str] = set(order_keys)
        self._applied: dict[str, AppliedExec] = {a.broker_exec_id: a for a in applied}

    # ------------------------------------------------------------------ ordine

    def has_order(self, client_order_id: str) -> bool:
        return client_order_id in self._orders

    def register_order(self, client_order_id: str) -> bool:
        """Înregistrează cheia; întoarce `False` dacă era deja înregistrată (10.2)."""
        if client_order_id in self._orders:
            return False
        self._orders.add(client_order_id)
        return True

    # ------------------------------------------------------------------ execuții

    def applied(self, broker_exec_id: str) -> AppliedExec | None:
        return self._applied.get(broker_exec_id)

    def mark_applied(self, entry: AppliedExec) -> None:
        existing = self._applied.get(entry.broker_exec_id)
        if existing is not None:
            raise ValueError(f"broker_exec_id {entry.broker_exec_id} este deja aplicat")
        self._applied[entry.broker_exec_id] = entry

    @property
    def applied_execs(self) -> Mapping[str, AppliedExec]:
        return dict(self._applied)
