"""
SQLite persistence backend for FL-EHDS governance layer.

Provides durable storage for permits, opt-out records, and audit logs.
Uses WAL journal mode for concurrent read/write access and thread-local
connections for thread safety.

Author: Fabio Liberti
"""

import sqlite3
import json
import threading
from pathlib import Path
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime


DEFAULT_DB_PATH = Path("data/governance.db")


class GovernanceDB:
    """Thread-safe SQLite backend for governance data."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize governance database.

        Args:
            db_path: Path to SQLite database file. Use ':memory:' for testing.
                     Defaults to 'data/governance.db'.
        """
        if db_path == ":memory:":
            self.db_path = ":memory:"
        else:
            self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._local = threading.local()
        self._init_schema()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local connection."""
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                detect_types=sqlite3.PARSE_DECLTYPES,
            )
            self._local.conn.row_factory = sqlite3.Row
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA foreign_keys=ON")
        return self._local.conn

    def _init_schema(self):
        """Create tables if they don't exist."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS permits (
                permit_id TEXT PRIMARY KEY,
                hdab_id TEXT NOT NULL,
                requester_id TEXT NOT NULL,
                purpose TEXT NOT NULL,
                data_categories TEXT NOT NULL DEFAULT '[]',
                data_sources TEXT NOT NULL DEFAULT '[]',
                member_states TEXT NOT NULL DEFAULT '[]',
                issued_at TEXT,
                valid_from TEXT,
                valid_until TEXT,
                status TEXT NOT NULL DEFAULT 'active',
                conditions TEXT DEFAULT '{}',
                metadata TEXT DEFAULT '{}',
                privacy_budget_total REAL,
                privacy_budget_used REAL DEFAULT 0,
                max_rounds INTEGER,
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS optout_records (
                record_id TEXT PRIMARY KEY,
                patient_id TEXT NOT NULL,
                opt_out_date TEXT,
                scope TEXT NOT NULL DEFAULT 'all',
                categories TEXT,
                purposes TEXT,
                member_state TEXT NOT NULL DEFAULT '',
                is_active INTEGER DEFAULT 1,
                metadata TEXT DEFAULT '{}',
                created_at TEXT DEFAULT (datetime('now')),
                updated_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_optout_patient
                ON optout_records(patient_id);
            CREATE INDEX IF NOT EXISTS idx_optout_active
                ON optout_records(is_active);

            CREATE TABLE IF NOT EXISTS audit_log (
                record_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                action TEXT NOT NULL,
                actor TEXT NOT NULL,
                permit_id TEXT,
                outcome TEXT,
                data_categories TEXT DEFAULT '[]',
                purpose TEXT,
                legal_basis TEXT,
                details TEXT DEFAULT '{}',
                client_ids TEXT DEFAULT '[]',
                round_number INTEGER,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                ON audit_log(timestamp);
            CREATE INDEX IF NOT EXISTS idx_audit_action
                ON audit_log(action);
            CREATE INDEX IF NOT EXISTS idx_audit_permit
                ON audit_log(permit_id);
        """)
        conn.commit()

    def close(self):
        """Close thread-local connection."""
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    # =========================================================================
    # PERMITS
    # =========================================================================

    def save_permit(self, permit) -> None:
        """Save or update a DataPermit."""
        conn = self._get_conn()
        data = permit.model_dump(mode="json")
        conn.execute(
            """INSERT OR REPLACE INTO permits
               (permit_id, hdab_id, requester_id, purpose, data_categories,
                data_sources, member_states, issued_at, valid_from, valid_until,
                status, conditions, metadata, privacy_budget_total,
                privacy_budget_used, max_rounds, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       datetime('now'))""",
            (
                data["permit_id"],
                data["hdab_id"],
                data["requester_id"],
                data["purpose"],
                json.dumps(data.get("data_categories", [])),
                json.dumps(data.get("data_sources", [])),
                json.dumps(data.get("member_states", [])),
                data.get("issued_at"),
                data.get("valid_from"),
                data.get("valid_until"),
                data["status"],
                json.dumps(data.get("conditions", {})),
                json.dumps(data.get("metadata", {})),
                data.get("privacy_budget_total"),
                data.get("privacy_budget_used", 0),
                data.get("max_rounds"),
            ),
        )
        conn.commit()

    def get_permit(self, permit_id: str) -> Optional[Dict[str, Any]]:
        """Get permit by ID. Returns raw dict (caller converts to DataPermit)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM permits WHERE permit_id = ?", (permit_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_permit_dict(row)

    def list_permits(
        self,
        requester_id: Optional[str] = None,
        status: Optional[str] = None,
        purpose: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """List permits with optional filters."""
        conn = self._get_conn()
        query = "SELECT * FROM permits WHERE 1=1"
        params: list = []
        if requester_id:
            query += " AND requester_id = ?"
            params.append(requester_id)
        if status:
            query += " AND status = ?"
            params.append(status)
        if purpose:
            query += " AND purpose = ?"
            params.append(purpose)
        rows = conn.execute(query, params).fetchall()
        return [self._row_to_permit_dict(r) for r in rows]

    def update_permit_status(
        self, permit_id: str, status: str, metadata_updates: Optional[Dict] = None
    ) -> bool:
        """Update permit status and optionally merge metadata."""
        conn = self._get_conn()
        if metadata_updates:
            row = conn.execute(
                "SELECT metadata FROM permits WHERE permit_id = ?", (permit_id,)
            ).fetchone()
            if row is None:
                return False
            existing = json.loads(row["metadata"])
            existing.update(metadata_updates)
            conn.execute(
                "UPDATE permits SET status = ?, metadata = ?, updated_at = datetime('now') WHERE permit_id = ?",
                (status, json.dumps(existing), permit_id),
            )
        else:
            conn.execute(
                "UPDATE permits SET status = ?, updated_at = datetime('now') WHERE permit_id = ?",
                (status, permit_id),
            )
        conn.commit()
        return conn.total_changes > 0

    def delete_permit(self, permit_id: str) -> bool:
        """Delete a permit."""
        conn = self._get_conn()
        conn.execute("DELETE FROM permits WHERE permit_id = ?", (permit_id,))
        conn.commit()
        return conn.total_changes > 0

    def _row_to_permit_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to a dict suitable for DataPermit construction."""
        return {
            "permit_id": row["permit_id"],
            "hdab_id": row["hdab_id"],
            "requester_id": row["requester_id"],
            "purpose": row["purpose"],
            "data_categories": json.loads(row["data_categories"]),
            "data_sources": json.loads(row["data_sources"]),
            "member_states": json.loads(row["member_states"]),
            "issued_at": row["issued_at"],
            "valid_from": row["valid_from"],
            "valid_until": row["valid_until"],
            "status": row["status"],
            "conditions": json.loads(row["conditions"]),
            "metadata": json.loads(row["metadata"]),
            "privacy_budget_total": row["privacy_budget_total"],
            "privacy_budget_used": row["privacy_budget_used"],
            "max_rounds": row["max_rounds"],
        }

    # =========================================================================
    # OPT-OUT RECORDS
    # =========================================================================

    def save_optout(self, record) -> None:
        """Save or update an OptOutRecord."""
        conn = self._get_conn()
        data = record.model_dump(mode="json")
        conn.execute(
            """INSERT OR REPLACE INTO optout_records
               (record_id, patient_id, opt_out_date, scope, categories,
                purposes, member_state, is_active, metadata, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
            (
                data["record_id"],
                data["patient_id"],
                data.get("opt_out_date"),
                data.get("scope", "all"),
                json.dumps(data.get("categories")) if data.get("categories") else None,
                json.dumps(data.get("purposes")) if data.get("purposes") else None,
                data.get("member_state", ""),
                1 if data.get("is_active", True) else 0,
                json.dumps(data.get("metadata", {})),
            ),
        )
        conn.commit()

    def is_opted_out(self, patient_id: str) -> bool:
        """Check if a patient has an active opt-out record."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT 1 FROM optout_records WHERE patient_id = ? AND is_active = 1 LIMIT 1",
            (patient_id,),
        ).fetchone()
        return row is not None

    def get_optout(self, record_id: str) -> Optional[Dict[str, Any]]:
        """Get opt-out record by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM optout_records WHERE record_id = ?", (record_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_optout_dict(row)

    def deactivate_optout(self, patient_id: str) -> bool:
        """Deactivate opt-out for a patient."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE optout_records SET is_active = 0, updated_at = datetime('now') WHERE patient_id = ? AND is_active = 1",
            (patient_id,),
        )
        conn.commit()
        return conn.total_changes > 0

    def list_active_optouts(self) -> List[Dict[str, Any]]:
        """List all active opt-out records."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT * FROM optout_records WHERE is_active = 1"
        ).fetchall()
        return [self._row_to_optout_dict(r) for r in rows]

    def count_optouts(self, active_only: bool = True) -> int:
        """Count opt-out records."""
        conn = self._get_conn()
        if active_only:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM optout_records WHERE is_active = 1"
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) as cnt FROM optout_records").fetchone()
        return row["cnt"]

    def _row_to_optout_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Convert SQLite row to dict for OptOutRecord construction."""
        return {
            "record_id": row["record_id"],
            "patient_id": row["patient_id"],
            "opt_out_date": row["opt_out_date"],
            "scope": row["scope"],
            "categories": json.loads(row["categories"]) if row["categories"] else None,
            "purposes": json.loads(row["purposes"]) if row["purposes"] else None,
            "member_state": row["member_state"],
            "is_active": bool(row["is_active"]),
            "metadata": json.loads(row["metadata"]) if row["metadata"] else {},
        }

    # =========================================================================
    # AUDIT LOG
    # =========================================================================

    def save_audit_records(self, records: list) -> None:
        """Save a batch of ComplianceRecord objects to the audit log."""
        conn = self._get_conn()
        for record in records:
            data = record.to_log_entry()
            conn.execute(
                """INSERT OR REPLACE INTO audit_log
                   (record_id, timestamp, action, actor, permit_id, outcome,
                    data_categories, purpose, legal_basis, details, client_ids,
                    round_number)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    data["record_id"],
                    data["timestamp"],
                    data["action"],
                    data["actor"],
                    data.get("permit_id"),
                    data.get("outcome"),
                    json.dumps(data.get("data_categories", [])),
                    data.get("purpose"),
                    data.get("legal_basis"),
                    json.dumps(data.get("details", {})),
                    json.dumps(getattr(record, "client_ids", [])),
                    getattr(record, "round_number", None),
                ),
            )
        conn.commit()

    def query_audit(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        action: Optional[str] = None,
        permit_id: Optional[str] = None,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        """Query audit log with filters."""
        conn = self._get_conn()
        query = "SELECT * FROM audit_log WHERE 1=1"
        params: list = []

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date.isoformat())
        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date.isoformat())
        if action:
            query += " AND action = ?"
            params.append(action)
        if permit_id:
            query += " AND permit_id = ?"
            params.append(permit_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [
            {
                "record_id": r["record_id"],
                "timestamp": r["timestamp"],
                "action": r["action"],
                "actor": r["actor"],
                "permit_id": r["permit_id"],
                "outcome": r["outcome"],
                "data_categories": json.loads(r["data_categories"]) if r["data_categories"] else [],
                "purpose": r["purpose"],
                "legal_basis": r["legal_basis"],
                "details": json.loads(r["details"]) if r["details"] else {},
                "client_ids": json.loads(r["client_ids"]) if r["client_ids"] else [],
                "round_number": r["round_number"],
            }
            for r in rows
        ]

    def save_permit_action(
        self, action: str, permit_id: str, details: Optional[Dict] = None
    ) -> None:
        """Save a simple permit audit action (used by PermitStore)."""
        conn = self._get_conn()
        conn.execute(
            """INSERT INTO audit_log
               (record_id, timestamp, action, actor, permit_id, outcome, details)
               VALUES (?, datetime('now'), ?, 'permit_store', ?, 'success', ?)""",
            (
                f"pa_{permit_id}_{action}_{datetime.utcnow().timestamp():.0f}",
                action,
                permit_id,
                json.dumps(details or {}),
            ),
        )
        conn.commit()
