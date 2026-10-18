"""The partner-scoped store (W1–W3).

Isolation is enforced here, in one place, and tested, not left to each service:

* Every partner-owned table has a ``partner_id`` column.
* Services never get a raw connection. They get a :class:`PartnerScope`, bound to one partner at
  construction, whose every read goes through :meth:`PartnerScope._select`, which always adds
  ``partner_id = ?``. No method takes a different partner id.
* Looking up another partner's object by id behaves exactly like looking up an id that does not
  exist (:class:`NotFound`), so existence does not leak, and the attempt is written to the audit log.
* Read paths are registered with ``@read_path`` so the access tests find and attack every one of
  them automatically, including ones added later.

What a partner owns: the lab data they imported (libraries, records, import reports), their
campaigns and everything a campaign produces (candidates, experiment requests, results, model
snapshots), agent runs, and their audit log. Platform-level operations (:class:`Platform`) return
aggregates or partner-free data only.
"""

from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

from lodestone import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS partners (
    partner_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    token_hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS libraries (
    row_id TEXT PRIMARY KEY,
    library_ref TEXT NOT NULL,
    partner_id TEXT NOT NULL REFERENCES partners(partner_id),
    source TEXT NOT NULL,
    source_library_id TEXT NOT NULL,
    system TEXT NOT NULL,
    synthesis_json TEXT,
    report_json TEXT NOT NULL,
    accepted INTEGER NOT NULL,
    rejected INTEGER NOT NULL,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (partner_id, library_ref)
);
CREATE TABLE IF NOT EXISTS records (
    row_id TEXT PRIMARY KEY,
    record_id TEXT NOT NULL,
    partner_id TEXT NOT NULL REFERENCES partners(partner_id),
    library_ref TEXT NOT NULL,
    system TEXT NOT NULL,
    record_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (partner_id, record_id)
);
CREATE TABLE IF NOT EXISTS campaigns (
    campaign_id TEXT PRIMARY KEY,
    partner_id TEXT NOT NULL REFERENCES partners(partner_id),
    name TEXT NOT NULL,
    system TEXT NOT NULL,
    spec_json TEXT NOT NULL,
    status TEXT NOT NULL,
    round INTEGER NOT NULL DEFAULT 0,
    created_by TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS candidates (
    candidate_id TEXT PRIMARY KEY,
    partner_id TEXT NOT NULL REFERENCES partners(partner_id),
    campaign_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    rank INTEGER NOT NULL,
    record_id TEXT NOT NULL,
    label TEXT NOT NULL,
    prediction_json TEXT,
    acquisition REAL,
    strategy TEXT NOT NULL,
    why TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS experiments (
    request_id TEXT PRIMARY KEY,
    partner_id TEXT NOT NULL REFERENCES partners(partner_id),
    campaign_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    record_id TEXT NOT NULL,
    label TEXT NOT NULL,
    status TEXT NOT NULL,
    due_round INTEGER NOT NULL,
    plan_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS results (
    result_id TEXT PRIMARY KEY,
    partner_id TEXT NOT NULL REFERENCES partners(partner_id),
    campaign_id TEXT NOT NULL,
    request_id TEXT NOT NULL,
    record_id TEXT NOT NULL,
    status TEXT NOT NULL,
    round INTEGER NOT NULL,
    result_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_snapshots (
    snapshot_id TEXT PRIMARY KEY,
    partner_id TEXT NOT NULL REFERENCES partners(partner_id),
    campaign_id TEXT NOT NULL,
    round INTEGER NOT NULL,
    meta_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id TEXT PRIMARY KEY,
    partner_id TEXT NOT NULL REFERENCES partners(partner_id),
    campaign_id TEXT,
    transcript_json TEXT NOT NULL,
    metrics_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS audit_log (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    partner_id TEXT NOT NULL,
    actor TEXT NOT NULL,
    action TEXT NOT NULL,
    object_type TEXT NOT NULL,
    object_id TEXT,
    outcome TEXT NOT NULL,
    detail TEXT
);
CREATE INDEX IF NOT EXISTS ix_records_p ON records(partner_id, system);
CREATE INDEX IF NOT EXISTS ix_candidates_p ON candidates(partner_id, campaign_id);
CREATE INDEX IF NOT EXISTS ix_experiments_p ON experiments(partner_id, campaign_id);
CREATE INDEX IF NOT EXISTS ix_results_p ON results(partner_id, campaign_id);
CREATE INDEX IF NOT EXISTS ix_audit_p ON audit_log(partner_id, event_id);
"""

# table → the id column callers look objects up by
PARTNER_TABLES = {
    "libraries": "library_ref",
    "records": "record_id",
    "campaigns": "campaign_id",
    "candidates": "candidate_id",
    "experiments": "request_id",
    "results": "result_id",
    "model_snapshots": "snapshot_id",
    "agent_runs": "run_id",
}
JSON_COLUMNS = {"synthesis_json", "report_json", "record_json", "spec_json", "prediction_json", "plan_json", "result_json", "meta_json", "transcript_json", "metrics_json"}


class NotFound(LookupError):
    """Raised for ids that do not exist *in this partner's scope* (other partners' ids included)."""


class AccessDenied(PermissionError):
    pass


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def _decode(row: sqlite3.Row) -> dict:
    out = dict(row)
    for k in list(out):
        if k in JSON_COLUMNS and out[k] is not None:
            out[k.removesuffix("_json")] = json.loads(out.pop(k))
    out.pop("row_id", None)
    return out


READ_PATHS: dict[str, dict] = {}


def read_path(*, by_id: str | None = None, by_campaign: bool = False) -> Callable:
    """Register a PartnerScope read method for the automated cross-partner access tests.

    by_id:       the method takes an id of this object type ("campaign", "record", ...)
    by_campaign: the method takes a campaign_id
    """

    def deco(fn):
        READ_PATHS[fn.__name__] = {"by_id": by_id, "by_campaign": by_campaign}
        return fn

    return deco


class Database:
    """Owns the SQLite connection. Hands out scopes, never the connection."""

    def __init__(self, path: str | Path | None = None):
        self.path = str(path or config.db_path())
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(SCHEMA)
        self._lock = threading.RLock()

    def _execute(self, sql: str, params: Iterable[Any] = ()) -> list[sqlite3.Row]:
        with self._lock:
            return self._conn.execute(sql, tuple(params)).fetchall()

    def _executemany(self, sql: str, rows: list[tuple]) -> None:
        with self._lock:
            self._conn.execute("BEGIN")
            try:
                self._conn.executemany(sql, rows)
                self._conn.execute("COMMIT")
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def scope(self, partner_id: str, actor: str = "user") -> "PartnerScope":
        return PartnerScope(self, partner_id, actor)

    def platform(self) -> "Platform":
        return Platform(self)

    def close(self):
        self._conn.close()


class Platform:
    """Platform-team operations. Returns aggregates or partner-free data only."""

    def __init__(self, db: Database):
        self._db = db

    def create_partner(self, name: str, partner_id: str | None = None, token: str | None = None) -> tuple[str, str]:
        partner_id = partner_id or new_id("partner")
        token = token or secrets.token_urlsafe(24)
        self._db._execute(
            "INSERT INTO partners(partner_id, name, token_hash, created_at) VALUES (?,?,?,?)",
            (partner_id, name, hash_token(token), _now()),
        )
        return partner_id, token

    def list_partners(self) -> list[dict]:
        return [dict(r) for r in self._db._execute("SELECT partner_id, name, created_at FROM partners ORDER BY created_at")]

    def partner_exists(self, partner_id: str) -> bool:
        return bool(self._db._execute("SELECT 1 FROM partners WHERE partner_id=?", (partner_id,)))

    def partner_for_token(self, token: str) -> str:
        rows = self._db._execute("SELECT partner_id FROM partners WHERE token_hash=?", (hash_token(token),))
        if not rows:
            raise AccessDenied("unknown API token")
        return rows[0]["partner_id"]

    def lab_queue_length(self) -> int:
        """Experiments waiting in the (shared) replay lab across all partners. A count only."""
        return self._db._execute("SELECT COUNT(*) AS n FROM experiments WHERE status='queued'")[0]["n"]

    def isolation_report(self) -> list[dict]:
        out = []
        for table in PARTNER_TABLES:
            for r in self._db._execute(f"SELECT partner_id, COUNT(*) AS n FROM {table} GROUP BY partner_id"):
                out.append({"table": table, "partner_id": r["partner_id"], "rows": r["n"]})
        return out

    def probe_ids(self, not_partner: str) -> dict[str, str]:
        """One object id per table owned by some *other* partner, used only by the live isolation probe."""
        out = {}
        for table, key in PARTNER_TABLES.items():
            rows = self._db._execute(
                f"SELECT {key} FROM {table} WHERE partner_id != ? AND {key} NOT IN (SELECT {key} FROM {table} WHERE partner_id = ?) LIMIT 1",
                (not_partner, not_partner),
            )
            if rows:
                out[table] = rows[0][0]
        return out


class PartnerScope:
    """Everything one partner's services can see or change."""

    def __init__(self, db: Database, partner_id: str, actor: str = "user"):
        if not db.platform().partner_exists(partner_id):
            raise AccessDenied(f"unknown partner {partner_id!r}")
        self._db = db
        self.partner_id = partner_id
        self.actor = actor

    def as_actor(self, actor: str) -> "PartnerScope":
        return PartnerScope(self._db, self.partner_id, actor)

    def platform_queue_length(self) -> int:
        return self._db.platform().lab_queue_length()

    # ------------------------------------------------------------------ the one place reads happen

    def _select(self, table: str, order_by: str | None = None, limit: int | None = None, **filters) -> list[dict]:
        if table not in PARTNER_TABLES and table != "audit_log":
            raise ValueError(f"{table} is not a partner table")
        where = ["partner_id = ?"]
        params: list[Any] = [self.partner_id]
        for col, val in filters.items():
            if val is None:
                continue
            if isinstance(val, (list, tuple, set)):
                val = list(val)
                if not val:
                    return []
                where.append(f"{col} IN ({','.join('?' * len(val))})")
                params.extend(val)
            else:
                where.append(f"{col} = ?")
                params.append(val)
        sql = f"SELECT * FROM {table} WHERE {' AND '.join(where)}"
        if order_by:
            sql += f" ORDER BY {order_by}"
        if limit:
            sql += f" LIMIT {int(limit)}"
        return [_decode(r) for r in self._db._execute(sql, params)]

    def _get_one(self, table: str, object_id: str) -> dict:
        key = PARTNER_TABLES[table]
        rows = self._select(table, **{key: object_id})
        if not rows:
            if self._db._execute(f"SELECT 1 FROM {table} WHERE {key} = ?", (object_id,)):
                self.log("read", table, object_id, outcome="denied", detail="object belongs to another partner")
            raise NotFound(f"{table.rstrip('s')} {object_id!r} not found")
        return rows[0]

    @staticmethod
    def _encode(row: dict) -> dict:
        row = dict(row)
        for k in list(row):
            if k + "_json" in JSON_COLUMNS and not k.endswith("_json"):
                row[k + "_json"] = json.dumps(row.pop(k), default=str)
        return row

    def _insert(self, table: str, row: dict) -> dict:
        row = self._encode(row)
        row["partner_id"] = self.partner_id  # never trust a caller-supplied partner id
        cols = ",".join(row)
        self._db._execute(f"INSERT INTO {table}({cols}) VALUES ({','.join('?' * len(row))})", row.values())
        return row

    def _update(self, table: str, object_id: str, **changes) -> None:
        key = PARTNER_TABLES[table]
        self._get_one(table, object_id)  # scope check first
        changes = self._encode(changes)
        sets = ", ".join(f"{k} = ?" for k in changes)
        self._db._execute(f"UPDATE {table} SET {sets} WHERE {key} = ? AND partner_id = ?", [*changes.values(), object_id, self.partner_id])

    # ------------------------------------------------------------------------------------ audit

    def log(self, action: str, object_type: str, object_id: str | None = None, outcome: str = "ok", detail: str | None = None) -> None:
        self._db._execute(
            "INSERT INTO audit_log(ts, partner_id, actor, action, object_type, object_id, outcome, detail) VALUES (?,?,?,?,?,?,?,?)",
            (_now(), self.partner_id, self.actor, action, object_type, object_id, outcome, detail),
        )

    @read_path()
    def list_audit(self, limit: int = 300) -> list[dict]:
        return self._select("audit_log", order_by="event_id DESC", limit=limit)

    def partner(self) -> dict:
        r = self._db._execute("SELECT partner_id, name FROM partners WHERE partner_id=?", (self.partner_id,))
        return dict(r[0])

    # ---------------------------------------------------------------------------- lab data

    def add_library(self, library_ref: str, source: str, source_library_id: str, system: str, synthesis: dict | None, report: dict, records: list[dict]) -> dict:
        """Store one imported library, its accepted records and its import report, atomically enough for a prototype."""
        if self._select("libraries", library_ref=library_ref):
            raise ValueError(f"library {library_ref} is already imported in this workspace")
        self._insert("libraries", {
            "row_id": uuid.uuid4().hex, "library_ref": library_ref, "source": source, "source_library_id": source_library_id,
            "system": system, "synthesis": synthesis, "report": report, "accepted": report["accepted"],
            "rejected": report["rejected"], "created_by": self.actor, "created_at": _now(),
        })
        now = _now()
        self._db._executemany(
            "INSERT INTO records(row_id, record_id, partner_id, library_ref, system, record_json, created_at) VALUES (?,?,?,?,?,?,?)",
            [(uuid.uuid4().hex, r["record_id"], self.partner_id, library_ref, system, json.dumps(r), now) for r in records],
        )
        self.log("import", "library", library_ref, detail=f"{report['accepted']} accepted, {report['rejected']} rejected")
        return self.get_library(library_ref)

    @read_path(by_id="library")
    def get_library(self, library_ref: str) -> dict:
        return self._get_one("libraries", library_ref)

    @read_path()
    def list_libraries(self, system: str | None = None) -> list[dict]:
        return self._select("libraries", order_by="system, source_library_id", system=system)

    @read_path(by_id="record")
    def get_record(self, record_id: str) -> dict:
        return self._get_one("records", record_id)["record"]

    @read_path()
    def list_records(self, system: str | None = None, library_ref: str | None = None) -> list[dict]:
        return [r["record"] for r in self._select("records", order_by="record_id", system=system, library_ref=library_ref)]

    def existing_source_ids(self) -> set[str]:
        return {r["record"]["provenance"]["source_sample_id"] for r in self._select("records")}

    def systems(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self._db._execute("SELECT system, COUNT(*) AS n FROM records WHERE partner_id=? GROUP BY system", (self.partner_id,)):
            out[r["system"]] = r["n"]
        return out

    # ---------------------------------------------------------------------------- campaigns

    def create_campaign(self, spec: dict) -> dict:
        cid = new_id("cmp")
        self._insert("campaigns", {
            "campaign_id": cid, "name": spec["name"], "system": spec["system"], "spec": spec,
            "status": "active", "round": 0, "created_by": self.actor, "created_at": _now(),
        })
        self.log("create", "campaign", cid, detail=spec["name"])
        return self.get_campaign(cid)

    @read_path(by_id="campaign")
    def get_campaign(self, campaign_id: str) -> dict:
        return self._get_one("campaigns", campaign_id)

    @read_path()
    def list_campaigns(self) -> list[dict]:
        return self._select("campaigns", order_by="created_at DESC")

    def update_campaign(self, campaign_id: str, **changes) -> None:
        self._update("campaigns", campaign_id, **changes)

    def add_candidates(self, rows: list[dict]) -> None:
        for r in rows:
            self._insert("candidates", {**r, "created_at": _now()})
        if rows:
            self.log("propose", "candidates", rows[0]["campaign_id"], detail=f"{len(rows)} in round {rows[0]['round']}")

    @read_path(by_campaign=True)
    def list_candidates(self, campaign_id: str, status: str | list[str] | None = None, round: int | None = None) -> list[dict]:
        self.get_campaign(campaign_id)
        return self._select("candidates", order_by="round, rank", campaign_id=campaign_id, status=status, round=round)

    @read_path(by_id="candidate")
    def get_candidate(self, candidate_id: str) -> dict:
        return self._get_one("candidates", candidate_id)

    def update_candidate(self, candidate_id: str, **changes) -> None:
        self._update("candidates", candidate_id, **changes)

    def add_experiment(self, row: dict) -> None:
        self._insert("experiments", {**row, "created_at": _now()})

    @read_path(by_campaign=True)
    def list_experiments(self, campaign_id: str, status: str | list[str] | None = None) -> list[dict]:
        self.get_campaign(campaign_id)
        return self._select("experiments", order_by="round, request_id", campaign_id=campaign_id, status=status)

    @read_path(by_id="experiment")
    def get_experiment(self, request_id: str) -> dict:
        return self._get_one("experiments", request_id)

    def update_experiment(self, request_id: str, **changes) -> None:
        self._update("experiments", request_id, **changes)

    def add_result(self, row: dict) -> None:
        self._insert("results", {**row, "created_at": _now()})

    @read_path(by_campaign=True)
    def list_results(self, campaign_id: str, status: str | None = None) -> list[dict]:
        self.get_campaign(campaign_id)
        return self._select("results", order_by="round, created_at, result_id", campaign_id=campaign_id, status=status)

    @read_path(by_id="result")
    def get_result(self, result_id: str) -> dict:
        return self._get_one("results", result_id)

    def add_model_snapshot(self, campaign_id: str, round: int, meta: dict) -> None:
        self._insert("model_snapshots", {"snapshot_id": new_id("mdl"), "campaign_id": campaign_id, "round": round, "meta": meta, "created_at": _now()})

    @read_path(by_campaign=True)
    def list_model_snapshots(self, campaign_id: str) -> list[dict]:
        self.get_campaign(campaign_id)
        return self._select("model_snapshots", order_by="round", campaign_id=campaign_id)

    # ---------------------------------------------------------------------------- agent runs

    def save_agent_run(self, campaign_id: str | None, transcript: list, metrics: dict) -> str:
        run_id = new_id("run")
        self._insert("agent_runs", {"run_id": run_id, "campaign_id": campaign_id, "transcript": transcript, "metrics": metrics, "created_at": _now()})
        self.log("create", "agent_run", run_id)
        return run_id

    @read_path(by_id="agent_run")
    def get_agent_run(self, run_id: str) -> dict:
        return self._get_one("agent_runs", run_id)

    @read_path()
    def list_agent_runs(self) -> list[dict]:
        return self._select("agent_runs", order_by="created_at DESC")
