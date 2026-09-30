# ============================================================
# Alert and Flagging Service
# Financial Fraud Detection System
#
# Uses PostgreSQL for alert storage
# Supports migration from existing SQLite alerts
# ============================================================

import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
import sqlalchemy as sa
from sqlalchemy import (
    Integer,
    MetaData,
    Table,
    Column,
    BigInteger,
    String,
    Float,
    DateTime,
    select,
    insert,
    update,
    delete,
    func,
    text,
)
from sqlalchemy.engine import Engine

logger = logging.getLogger("fraud_alert_service")

# ============================================================
# Project Paths & Environment Setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "backend" / "data"
DATABASE_FILE = DATA_DIR / "fraud_alerts.db"

# Ensure data directory exists
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Load environment variables from .env file
load_dotenv(PROJECT_ROOT / ".env")

# ============================================================
# Database Engine & URL Resolver
# ============================================================

def get_database_url() -> str:
    """
    Resolves the configured PostgreSQL connection string.

    Priority:
    1. DATABASE_URL from .env
    2. PostgreSQL connection from individual DB_* variables

    PostgreSQL is required for the active application.
    """

    raw_url = os.getenv("DATABASE_URL", "").strip()

    if raw_url:
        # Standardize PostgreSQL dialect for SQLAlchemy + psycopg2
        if raw_url.startswith("postgres://"):
            return raw_url.replace(
                "postgres://",
                "postgresql+psycopg2://",
                1,
            )

        if raw_url.startswith("postgresql://"):
            return raw_url.replace(
                "postgresql://",
                "postgresql+psycopg2://",
                1,
            )

        if raw_url.startswith("postgresql+psycopg2://"):
            return raw_url

        return raw_url

    # Check individual PostgreSQL components
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")

    if db_user and db_password and db_name:
        return (
            f"postgresql+psycopg2://"
            f"{db_user}:{db_password}@"
            f"{db_host}:{db_port}/{db_name}"
        )

    raise RuntimeError(
        "PostgreSQL configuration is missing. "
        "Please configure DATABASE_URL or DB_USER, DB_PASSWORD, "
        "DB_HOST, DB_PORT, and DB_NAME in the .env file."
    )


_engine: Optional[Engine] = None
_last_pg_error: Optional[str] = None


def get_engine() -> Engine:
    """
    Returns a singleton SQLAlchemy PostgreSQL engine.

    PostgreSQL connection errors are raised instead of silently
    falling back to SQLite, so database problems are visible.
    """

    global _engine, _last_pg_error

    if _engine is None:
        db_url = get_database_url()

        if not db_url.startswith("postgresql"):
            raise RuntimeError(
                "Only PostgreSQL is supported by the active alert service."
            )

        try:
            pg_engine = sa.create_engine(
                db_url,
                pool_pre_ping=True,
                pool_size=5,
                max_overflow=10,
                connect_args={"connect_timeout": 5},
            )

            # Test PostgreSQL connection
            with pg_engine.connect() as test_conn:
                test_conn.execute(text("SELECT 1"))

            _engine = pg_engine
            _last_pg_error = None

            logger.info("PostgreSQL connection established successfully.")

        except Exception as error:
            _last_pg_error = str(error)

            logger.error(
                "PostgreSQL connection failed: %s",
                error,
            )

            raise RuntimeError(
                f"PostgreSQL connection failed: {error}"
            ) from error

    return _engine


def reset_engine():
    """
    Resets the singleton PostgreSQL engine.
    Useful when connection settings change.
    """
    global _engine, _last_pg_error

    if _engine is not None:
        _engine.dispose()
        _engine = None

    _last_pg_error = None


# ============================================================
# PostgreSQL Schema Definition
# ============================================================

metadata = MetaData()

alerts_table = Table(
    "fraud_alerts",
    metadata,

    Column(
        "id",
        BigInteger,
        primary_key=True,
        autoincrement=True,
    ),

    Column(
        "transaction_id",
        String(255),
        nullable=False,
        unique=True,
        index=True,
    ),

    Column(
        "fraud_probability",
        Float,
        nullable=False,
    ),

    Column(
        "prediction",
        String(50),
        nullable=False,
    ),

    Column(
        "risk_score",
        Float,
        nullable=False,
    ),

    Column(
        "risk_level",
        String(20),
        nullable=False,
    ),

    Column(
        "status",
        String(20),
        nullable=False,
        default="New",
    ),

    Column(
        "created_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    ),

    Column(
        "updated_at",
        DateTime(timezone=True),
        nullable=False,
        server_default=sa.text("CURRENT_TIMESTAMP"),
    ),
)


# ============================================================
# Database Initialization
# ============================================================

def initialize_database():
    """
    Verifies the PostgreSQL connection and creates the
    fraud_alerts table if it does not already exist.
    Also ensures unique index on transaction_id to prevent duplicates.
    """

    try:
        engine = get_engine()
        metadata.create_all(engine)

        with engine.begin() as conn:
            conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_fraud_alerts_tx ON fraud_alerts (transaction_id);"))

        logger.info(
            "PostgreSQL database initialized successfully."
        )

    except Exception as error:
        raise RuntimeError(
            f"Database initialization failed: {error}"
        ) from error


# ============================================================
# Database Connection Test
# ============================================================

def test_connection() -> Dict[str, Any]:
    """
    Verifies the active PostgreSQL connection.

    Returns connection status and database information.
    """

    try:
        engine = get_engine()

        with engine.connect() as connection:

            result = connection.execute(
                text("SELECT 1")
            ).scalar()

            database_name = connection.execute(
                text("SELECT current_database()")
            ).scalar()

            database_user = connection.execute(
                text("SELECT current_user")
            ).scalar()

            server_port = connection.execute(
                text("SHOW port")
            ).scalar()

            url_display = str(
                engine.url.render_as_string(
                    hide_password=True
                )
            )

            return {
                "status": "connected"
                if result == 1
                else "unexpected_result",

                "database_type": engine.dialect.name,

                "is_postgresql": (
                    engine.dialect.name == "postgresql"
                ),

                "database_name": database_name,

                "database_user": database_user,

                "server_port": server_port,

                "url": url_display,
            }

    except Exception as error:

        return {
            "status": "error",
            "database_type": "postgresql",
            "is_postgresql": False,
            "error": str(error),
        }


# ============================================================
# Create Alert
# ============================================================

def create_alert(
    transaction_id: str,
    fraud_probability: float,
    prediction: str,
    risk_score: float,
    risk_level: str,
    status: str = "New",
) -> Dict[str, Any]:
    """
    Creates or updates an alert for a suspicious transaction.
    Initial status upon prediction is 'New'.
    Guarantees deduplication: if transaction_id already exists,
    its risk telemetry is refreshed without duplicating records.
    """

    if prediction != "Suspicious":
        return {
            "alert_created": False,
            "message": "Transaction does not require an alert.",
        }

    initialize_database()

    current_time = datetime.now(timezone.utc)

    engine = get_engine()

    try:
        with engine.begin() as connection:
            # Check if an alert for this transaction already exists
            existing = connection.execute(
                select(alerts_table.c.id, alerts_table.c.status).where(
                    alerts_table.c.transaction_id == transaction_id
                )
            ).fetchone()

            if existing:
                alert_id, existing_status = existing
                target_status = status or "New"
                connection.execute(
                    update(alerts_table)
                    .where(alerts_table.c.id == alert_id)
                    .values(
                        fraud_probability=fraud_probability,
                        prediction=prediction,
                        risk_score=risk_score,
                        risk_level=risk_level,
                        status=target_status,
                        updated_at=current_time,
                    )
                )
                return {
                    "alert_created": False,
                    "alert_updated": True,
                    "alert_id": alert_id,
                    "transaction_id": transaction_id,
                    "alert_status": target_status,
                    "message": f"Alert #{alert_id} for {transaction_id} updated with status '{target_status}'.",
                }

            target_status = status or "New"
            stmt = insert(alerts_table).values(
                transaction_id=transaction_id,
                fraud_probability=fraud_probability,
                prediction=prediction,
                risk_score=risk_score,
                risk_level=risk_level,
                status=target_status,
                created_at=current_time,
                updated_at=current_time,
            )
            result = connection.execute(stmt)
            alert_id = (
                result.inserted_primary_key[0]
                if result.inserted_primary_key
                else None
            )

        return {
            "alert_created": True,
            "alert_id": alert_id,
            "transaction_id": transaction_id,
            "alert_status": target_status,
            "message": f"Suspicious transaction flagged and recorded with status '{target_status}'.",
        }

    except Exception as error:
        raise RuntimeError(
            f"Failed to create alert: {error}"
        ) from error


def upsert_alerts_batch(
    alert_records: List[Dict[str, Any]]
) -> Dict[str, int]:
    """
    Bulk upserts a list of alerts into PostgreSQL.
    Transactions that already exist are updated with latest risk metrics.
    New transactions are inserted. Duplicates are never created.
    """
    if not alert_records:
        return {"total": 0, "inserted": 0, "updated": 0}

    initialize_database()
    current_time = datetime.now(timezone.utc)
    engine = get_engine()

    inserted_count = 0
    updated_count = 0

    try:
        with engine.begin() as connection:
            # Collect all candidate transaction IDs
            candidate_records = [
                r for r in alert_records
                if r.get("prediction") == "Suspicious" and r.get("transaction_id")
            ]
            if not candidate_records:
                return {"total": 0, "inserted": 0, "updated": 0}

            tx_ids = [r["transaction_id"] for r in candidate_records]

            # Find which transaction IDs already exist in the database
            existing_rows = connection.execute(
                select(alerts_table.c.id, alerts_table.c.transaction_id)
                .where(alerts_table.c.transaction_id.in_(tx_ids))
            ).fetchall()
            existing_map = {row[1]: row[0] for row in existing_rows}

            to_insert = []
            for r in candidate_records:
                tx_id = r["transaction_id"]
                if tx_id in existing_map:
                    # Refresh existing record without creating duplicate row
                    alert_id = existing_map[tx_id]
                    connection.execute(
                        update(alerts_table)
                        .where(alerts_table.c.id == alert_id)
                        .values(
                            fraud_probability=float(r.get("fraud_probability", 0.0)),
                            prediction=r.get("prediction", "Suspicious"),
                            risk_score=float(r.get("risk_score", 0.0)),
                            risk_level=str(r.get("risk_level", "High")),
                            updated_at=current_time,
                        )
                    )
                    updated_count += 1
                else:
                    to_insert.append({
                        "transaction_id": tx_id,
                        "fraud_probability": float(r.get("fraud_probability", 0.0)),
                        "prediction": r.get("prediction", "Suspicious"),
                        "risk_score": float(r.get("risk_score", 0.0)),
                        "risk_level": str(r.get("risk_level", "High")),
                        "status": "New",
                        "created_at": current_time,
                        "updated_at": current_time,
                    })
                    inserted_count += 1

            if to_insert:
                connection.execute(insert(alerts_table), to_insert)

        return {
            "total": inserted_count + updated_count,
            "inserted": inserted_count,
            "updated": updated_count,
        }

    except Exception as error:
        raise RuntimeError(
            f"Failed to bulk upsert alerts: {error}"
        ) from error


# ============================================================
# Get All Alerts
# ============================================================

def get_all_alerts(
    status: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Returns all alerts.

    Optional:
    status = New / Under Review / Resolved
    """

    initialize_database()

    engine = get_engine()

    try:

        stmt = select(alerts_table).order_by(
            alerts_table.c.created_at.desc()
        )

        if status:
            stmt = stmt.where(
                alerts_table.c.status == status
            )

        with engine.connect() as connection:

            rows = connection.execute(
                stmt
            ).mappings().all()

        return [dict(row) for row in rows]

    except Exception as error:

        raise RuntimeError(
            f"Failed to retrieve alerts: {error}"
        ) from error


# ============================================================
# Get Single Alert
# ============================================================

def get_alert(
    alert_id: int,
) -> Optional[Dict[str, Any]]:
    """
    Returns one alert using its alert ID.
    """

    initialize_database()

    engine = get_engine()

    try:

        stmt = select(alerts_table).where(
            alerts_table.c.id == alert_id
        )

        with engine.connect() as connection:

            row = connection.execute(
                stmt
            ).mappings().first()

        if row is None:
            return None

        return dict(row)

    except Exception as error:

        raise RuntimeError(
            f"Failed to retrieve alert: {error}"
        ) from error


# ============================================================
# Update Alert Status
# ============================================================

def update_alert_status(
    alert_id: int,
    status: str,
) -> Dict[str, Any]:
    """
    Updates an alert status.

    Allowed values:
    - New
    - Under Review
    - Resolved
    """

    allowed_statuses = [
        "New",
        "Under Review",
        "Resolved",
    ]

    if status not in allowed_statuses:

        raise ValueError(
            "Invalid alert status. "
            "Use: New, Under Review, or Resolved."
        )

    initialize_database()

    engine = get_engine()

    updated_time = datetime.now(timezone.utc)

    try:

        stmt = (
            update(alerts_table)
            .where(alerts_table.c.id == alert_id)
            .values(
                status=status,
                updated_at=updated_time,
            )
        )

        with engine.begin() as connection:

            result = connection.execute(stmt)

            if result.rowcount == 0:

                return {
                    "updated": False,
                    "message": "Alert not found.",
                }

        return {
            "updated": True,
            "alert_id": alert_id,
            "alert_status": status,
            "message": (
                "Alert status updated successfully."
            ),
        }

    except Exception as error:

        raise RuntimeError(
            f"Failed to update alert status: {error}"
        ) from error


# ============================================================
# Delete Alert
# ============================================================

def delete_alert(
    alert_id: int,
) -> Dict[str, Any]:
    """
    Deletes an alert by ID.
    """

    initialize_database()

    engine = get_engine()

    try:

        stmt = delete(alerts_table).where(
            alerts_table.c.id == alert_id
        )

        with engine.begin() as connection:

            result = connection.execute(stmt)

            if result.rowcount == 0:

                return {
                    "deleted": False,
                    "message": "Alert not found.",
                }

        return {
            "deleted": True,
            "alert_id": alert_id,
            "message": "Alert deleted successfully.",
        }

    except Exception as error:

        raise RuntimeError(
            f"Failed to delete alert: {error}"
        ) from error


# ============================================================
# Alert Summary
# ============================================================

def get_alert_summary() -> Dict[str, int]:
    """
    Returns alert counts for dashboard use.
    """

    initialize_database()

    engine = get_engine()

    try:

        with engine.connect() as connection:

            total_alerts = connection.execute(
                select(func.count()).select_from(
                    alerts_table
                )
            ).scalar() or 0

            new_alerts = connection.execute(
                select(func.count())
                .select_from(alerts_table)
                .where(
                    alerts_table.c.status == "New"
                )
            ).scalar() or 0

            under_review = connection.execute(
                select(func.count())
                .select_from(alerts_table)
                .where(
                    alerts_table.c.status
                    == "Under Review"
                )
            ).scalar() or 0

            resolved = connection.execute(
                select(func.count())
                .select_from(alerts_table)
                .where(
                    alerts_table.c.status == "Resolved"
                )
            ).scalar() or 0

        return {
            "total_alerts": total_alerts,
            "new_alerts": new_alerts,
            "under_review": under_review,
            "resolved": resolved,
        }

    except Exception as error:

        raise RuntimeError(
            f"Failed to create alert summary: {error}"
        ) from error


# ============================================================
# SQLite -> PostgreSQL Data Migration Utility
# ============================================================

def migrate_sqlite_to_postgres() -> Dict[str, Any]:
    """
    Migrates existing alerts from the old SQLite database
    to the PostgreSQL fraud_alerts table.
    """

    engine = get_engine()

    if engine.dialect.name != "postgresql":

        return {
            "migrated": False,
            "message": (
                "Target database is not PostgreSQL."
            ),
        }

    if not DATABASE_FILE.exists():

        return {
            "migrated": False,
            "message": (
                f"SQLite database file not found at "
                f"{DATABASE_FILE}."
            ),
        }

    # --------------------------------------------------------
    # Define the OLD SQLite table structure
    # --------------------------------------------------------

    sqlite_metadata = MetaData()

    sqlite_alerts_table = Table(
        "alerts",
        sqlite_metadata,

        Column(
            "id",
            Integer,
            primary_key=True,
        ),

        Column(
            "transaction_id",
            String(100),
            nullable=False,
        ),

        Column(
            "fraud_probability",
            Float,
            nullable=False,
        ),

        Column(
            "prediction",
            String(50),
            nullable=False,
        ),

        Column(
            "risk_score",
            Float,
            nullable=False,
        ),

        Column(
            "risk_level",
            String(50),
            nullable=False,
        ),

        Column(
            "alert_status",
            String(50),
            nullable=False,
        ),

        Column(
            "created_at",
            String(100),
            nullable=False,
        ),

        Column(
            "updated_at",
            String(100),
            nullable=False,
        ),
    )

    sqlite_engine = sa.create_engine(
        f"sqlite:///{DATABASE_FILE}"
    )

    try:

        # ----------------------------------------------------
        # Read alerts from SQLite
        # ----------------------------------------------------

        with sqlite_engine.connect() as sqlite_conn:

            rows = sqlite_conn.execute(
                select(
                    sqlite_alerts_table.c.transaction_id,
                    sqlite_alerts_table.c.fraud_probability,
                    sqlite_alerts_table.c.prediction,
                    sqlite_alerts_table.c.risk_score,
                    sqlite_alerts_table.c.risk_level,
                    sqlite_alerts_table.c.alert_status,
                    sqlite_alerts_table.c.created_at,
                    sqlite_alerts_table.c.updated_at,
                )
            ).mappings().all()

        if not rows:

            return {
                "migrated": True,
                "migrated_count": 0,
                "message": "No rows to migrate.",
            }

        # ----------------------------------------------------
        # Insert into PostgreSQL
        # ----------------------------------------------------

        migrated_count = 0

        with engine.begin() as pg_conn:

            for row in rows:

                pg_conn.execute(
                    insert(alerts_table).values(
                        transaction_id=row[
                            "transaction_id"
                        ],

                        fraud_probability=row[
                            "fraud_probability"
                        ],

                        prediction=row[
                            "prediction"
                        ],

                        risk_score=row[
                            "risk_score"
                        ],

                        risk_level=row[
                            "risk_level"
                        ],

                        status=row[
                            "alert_status"
                        ],

                        created_at=_parse_datetime(
                            row["created_at"]
                        ),

                        updated_at=_parse_datetime(
                            row["updated_at"]
                        ),
                    )
                )

                migrated_count += 1

        return {
            "migrated": True,
            "migrated_count": migrated_count,
            "message": (
                f"Successfully migrated "
                f"{migrated_count} alerts "
                f"from SQLite to PostgreSQL."
            ),
        }

    except Exception as error:

        raise RuntimeError(
            f"Migration failed: {error}"
        ) from error

    finally:

        sqlite_engine.dispose()


# ============================================================
# Date Conversion Helper
# ============================================================

def _parse_datetime(
    value: Any,
) -> datetime:
    """
    Converts an old SQLite timestamp string
    into a timezone-aware datetime.
    """

    if isinstance(value, datetime):

        if value.tzinfo is None:
            return value.replace(
                tzinfo=timezone.utc
            )

        return value

    if not value:

        return datetime.now(timezone.utc)

    value = str(value)

    try:

        parsed = datetime.fromisoformat(
            value.replace(
                "Z",
                "+00:00",
            )
        )

        if parsed.tzinfo is None:
            parsed = parsed.replace(
                tzinfo=timezone.utc
            )

        return parsed

    except ValueError:

        return datetime.now(timezone.utc)


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    try:

        print(
            "Testing PostgreSQL Database Connection..."
        )

        conn_info = test_connection()

        print(
            "Connection Info:",
            conn_info,
        )

        initialize_database()

        print(
            "PostgreSQL alert database "
            "initialized successfully."
        )

        print(
            "\nCurrent alert summary:"
        )

        print(
            get_alert_summary()
        )

    except Exception as error:

        print(
            f"\nAlert service test failed:\n{error}"
        )