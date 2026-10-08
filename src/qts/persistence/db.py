"""Conexiune SQLite cu durabilitate maximă și migrații versionate.

- `journal_mode=WAL` și `synchronous=FULL`: un commit confirmat supraviețuiește unei căderi.
- `isolation_level=None`: tranzacțiile sunt controlate explicit (`BEGIN IMMEDIATE`), astfel încât
  jurnalul și proiecțiile se scriu atomic împreună (design: Persistență).
- Migrațiile sunt doar adăugate, niciodată modificate după publicare.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Final

MIGRATIONS: Final[tuple[str, ...]] = (
    # 1: jurnalul append-only cu lanț hash (Req 24.4, 26.1)
    """
    CREATE TABLE journal (
        seq INTEGER PRIMARY KEY,
        ts TEXT NOT NULL,
        type TEXT NOT NULL,
        correlation_id TEXT NOT NULL,
        component TEXT NOT NULL,
        component_version TEXT NOT NULL,
        actor TEXT NOT NULL,
        outcome TEXT NOT NULL,
        payload TEXT NOT NULL,
        prev_hash TEXT NOT NULL,
        hash TEXT NOT NULL UNIQUE
    );
    CREATE INDEX journal_correlation ON journal(correlation_id);
    CREATE INDEX journal_type ON journal(type);
    CREATE TRIGGER journal_no_update BEFORE UPDATE ON journal
        BEGIN SELECT RAISE(ABORT, 'journal este append-only'); END;
    CREATE TRIGGER journal_no_delete BEFORE DELETE ON journal
        BEGIN SELECT RAISE(ABORT, 'journal este append-only'); END;
    """,
    # 2: Recovery_Point — checkpoint-uri append-only (Req 26.1, 26.2)
    """
    CREATE TABLE checkpoints (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        journal_seq INTEGER NOT NULL,
        journal_hash TEXT NOT NULL,
        projections_hash TEXT NOT NULL,
        ts TEXT NOT NULL
    );
    CREATE INDEX checkpoints_journal_seq ON checkpoints(journal_seq);
    CREATE TRIGGER checkpoints_no_update BEFORE UPDATE ON checkpoints
        BEGIN SELECT RAISE(ABORT, 'checkpoints este append-only'); END;
    CREATE TRIGGER checkpoints_no_delete BEFORE DELETE ON checkpoints
        BEGIN SELECT RAISE(ABORT, 'checkpoints este append-only'); END;
    """,
    # 3: coada operațională de alerte (Req 25.4, 25.6, 26.6). Spre deosebire de jurnal și
    # checkpoints, `alerts` NU este append-only: este o coadă de livrare, nu o înregistrare de
    # audit, deci starea livrării (status, attempts, last_error) se actualizează pe loc. Trasarea
    # pentru audit a alertelor rămâne în jurnalul append-only (Req 25.6 cere Audit_Record separat).
    """
    CREATE TABLE alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        severity TEXT NOT NULL,
        component TEXT NOT NULL,
        code TEXT NOT NULL,
        detail TEXT NOT NULL,
        ts TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        attempts INTEGER NOT NULL DEFAULT 0,
        last_error TEXT
    );
    CREATE INDEX alerts_status ON alerts(status);
    """,
    # 4: cercetare — partiții de date și registrul OOS cu consum unic (Req 18.1–18.6).
    # `datasets` înregistrează fiecare partiție cronologică (Development_Set / Out_Of_Sample_Set)
    # cu limitele temporale, identificatorul și rolul ei (Req 18.5). `oos_registry` ține evidența
    # rezervării și a consumului unic al fiecărui Out_Of_Sample_Set (Req 18.3, 18.6): rândul este
    # creat „reserved”, trece o singură dată în „consumed”, iar o reclasificare (Req 18.4) îl
    # marchează „reclassified” fără să mai permită consumul. Tabelele sunt append-only pentru
    # conținutul definitoriu; starea de consum a OOS se actualizează controlat prin tranziții.
    """
    CREATE TABLE datasets (
        dataset_id TEXT PRIMARY KEY,
        partition_hash TEXT NOT NULL UNIQUE,
        role TEXT NOT NULL,
        label TEXT NOT NULL,
        start_ts TEXT NOT NULL,
        end_ts TEXT NOT NULL,
        instruments TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX datasets_role ON datasets(role);
    CREATE TRIGGER datasets_no_update BEFORE UPDATE ON datasets
        BEGIN SELECT RAISE(ABORT, 'datasets este imuabil'); END;
    CREATE TRIGGER datasets_no_delete BEFORE DELETE ON datasets
        BEGIN SELECT RAISE(ABORT, 'datasets este imuabil'); END;

    CREATE TABLE oos_registry (
        dataset_id TEXT PRIMARY KEY REFERENCES datasets(dataset_id),
        decision_id TEXT NOT NULL,
        preregistration_hash TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'reserved',
        reserved_at TEXT NOT NULL,
        consumed_at TEXT,
        evaluation_hash TEXT,
        reclassified_at TEXT,
        reclassified_reason TEXT
    );
    CREATE INDEX oos_registry_decision ON oos_registry(decision_id);
    CREATE INDEX oos_registry_status ON oos_registry(status);
    CREATE TRIGGER oos_registry_no_delete BEFORE DELETE ON oos_registry
        BEGIN SELECT RAISE(ABORT, 'oos_registry este append-only'); END;
    """,
    # 5: registrul Open_Decision și aprobările (Req 30.1–30.9, 22.1, 22.2, 22.8, 3.5, 3.6).
    # `decisions` înregistrează fiecare Open_Decision cu opțiunile, criteriile măsurabile,
    # responsabilul, termenul, domeniile afectate și etapa dependentă blocată (Req 30.2).
    # Conținutul definitoriu este imuabil: `decision_id` = SHA-256 peste el, deci orice
    # modificare l-ar schimba.
    # Starea deciziei se reflectă prin existența unui rând în `approvals` (append-only): o decizie
    # este „aprobată” exact când are o aprobare, altfel rămâne „deschisă” și blochează etapele
    # dependente (Req 30.5, 30.9). `approvals` păstrează alegerea, dovezile, consecințele și
    # aprobatorul (Req 30.4, 22.2, 22.8). Ambele tabele sunt imuabile după scriere.
    """
    CREATE TABLE decisions (
        decision_id TEXT PRIMARY KEY,
        content_hash TEXT NOT NULL UNIQUE,
        topic TEXT NOT NULL UNIQUE,
        title TEXT NOT NULL,
        options TEXT NOT NULL,
        criteria TEXT NOT NULL,
        responsible TEXT NOT NULL,
        deadline TEXT NOT NULL,
        affected_domains TEXT NOT NULL,
        blocked_stage TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX decisions_topic ON decisions(topic);
    CREATE INDEX decisions_blocked_stage ON decisions(blocked_stage);
    CREATE TRIGGER decisions_no_update BEFORE UPDATE ON decisions
        BEGIN SELECT RAISE(ABORT, 'decisions este imuabil'); END;
    CREATE TRIGGER decisions_no_delete BEFORE DELETE ON decisions
        BEGIN SELECT RAISE(ABORT, 'decisions este imuabil'); END;

    CREATE TABLE approvals (
        decision_id TEXT PRIMARY KEY REFERENCES decisions(decision_id),
        chosen_option TEXT NOT NULL,
        criteria TEXT NOT NULL,
        evidence TEXT NOT NULL,
        consequences TEXT NOT NULL,
        approver TEXT NOT NULL,
        approved_at TEXT NOT NULL
    );
    CREATE TRIGGER approvals_no_update BEFORE UPDATE ON approvals
        BEGIN SELECT RAISE(ABORT, 'approvals este imuabil'); END;
    CREATE TRIGGER approvals_no_delete BEFORE DELETE ON approvals
        BEGIN SELECT RAISE(ABORT, 'approvals este imuabil'); END;
    """,
)


def open_db(path: Path | str) -> sqlite3.Connection:
    is_memory = str(path) == ":memory:"
    if not is_memory:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    if not is_memory:
        mode = conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            conn.close()
            raise RuntimeError(f"SQLite nu a activat WAL (mod: {mode})")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    migrate(conn)
    return conn


def migrate(conn: sqlite3.Connection) -> int:
    """Aplică migrațiile lipsă; întoarce versiunea finală a schemei."""
    current = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if current > len(MIGRATIONS):
        raise RuntimeError(
            f"baza de date are schema {current}, mai nouă decât codul ({len(MIGRATIONS)})"
        )
    for version in range(current + 1, len(MIGRATIONS) + 1):
        script = MIGRATIONS[version - 1]
        try:
            conn.executescript(
                f"BEGIN IMMEDIATE;\n{script}\nPRAGMA user_version={version};\nCOMMIT;"
            )
        except sqlite3.Error:
            if conn.in_transaction:
                conn.execute("ROLLBACK")
            raise
    return len(MIGRATIONS)
