# ============================================================
# Alert and Flagging Service
# Financial Fraud Detection System
#
# Supports PostgreSQL (production) with automatic SQLite fallback
# ============================================================

import os
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
import sqlalchemy as sa
from sqlalchemy import (
    MetaData,
    Table,
    Column,
    Integer,
    String,
    Float,
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

# Load environment variables from .env file in project root or backend dir
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv(DATA_DIR.parent / ".env")

# ============================================================
# Database Engine & URL Resolver
# ============================================================

def get_database_url() -> str:
    """
    Resolves the configured database connection string.

    Priority:
    1. Explicit DATABASE_URL in environment (.env).
    2. Composed PostgreSQL URL from DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME.
    3. Fallback to local SQLite database in backend/data/fraud_alerts.db.
    """
    raw_url = os.getenv("DATABASE_URL", "").strip()

    if raw_url:
        # Standardize PostgreSQL dialect for SQLAlchemy + psycopg2
        if raw_url.startswith("postgres://"):
            return raw_url.replace("postgres://", "postgresql+psycopg2://", 1)
        elif raw_url.startswith("postgresql://"):
            return raw_url.replace("postgresql://", "postgresql+psycopg2://", 1)
        return raw_url

    # Check individual components
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME")

    if db_user and db_password and db_name:
        return f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    # Default fallback to SQLite
    return f"sqlite:///{DATABASE_FILE}"


_engine: Optional[Engine] = None
_last_pg_error: Optional[str] = None


def get_engine() -> Engine:
    """
    Returns a singleton SQLAlchemy engine instance.
    If PostgreSQL is configured but unreachable (e.g. service not started yet),
    gracefully falls back to the local SQLite database so the application never crashes.
    """
    global _engine, _last_pg_error

    if _engine is None:
        db_url = get_database_url()

        if db_url.startswith("sqlite"):
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            _engine = sa.create_engine(
                db_url,
                connect_args={"check_same_thread": False},
            )
        else:
            # Attempt connection to PostgreSQL
            try:
                pg_engine = sa.create_engine(
                    db_url,
                    pool_pre_ping=True,
                    pool_size=5,
                    max_overflow=10,
                    connect_args={"connect_timeout": 3},
                )
                with pg_engine.connect() as test_conn:
                    test_conn.execute(text("SELECT 1"))

                _engine = pg_engine
                _last_pg_error = None

            except Exception as error:
                _last_pg_error = str(error)
                logger.warning(
                    "PostgreSQL is currently unreachable: %s. "
                    "Falling back to local SQLite database (%s).",
                    error,
                    DATABASE_FILE,
                )
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                _engine = sa.create_engine(
                    f"sqlite:///{DATABASE_FILE}",
                    connect_args={"check_same_thread": False},
                )

    return _engine


def reset_engine():
    """
    Resets the singleton engine (useful when connection settings change or PostgreSQL starts).
    """
    global _engine, _last_pg_error
    if _engine is not None:
        _engine.dispose()
        _engine = None
    _last_pg_error = None


# ============================================================
# Schema Definition
# ============================================================

metadata = MetaData()

alerts_table = Table(
    "alerts",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("transaction_id", String(100), nullable=False),
    Column("fraud_probability", Float, nullable=False),
    Column("prediction", String(50), nullable=False),
    Column("risk_score", Float, nullable=False),
    Column("risk_level", String(50), nullable=False),
    Column("alert_status", String(50), nullable=False, default="New"),
    Column("created_at", String(100), nullable=False),
    Column("updated_at", String(100), nullable=False),
)


# ============================================================
# Database Initialization
# ============================================================

def initialize_database():
    """
    Creates the alert database and alerts table if they do not already exist.
    Compatible with both PostgreSQL and SQLite.
    """
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        engine = get_engine()
        metadata.create_all(engine)
    except Exception as error:
        raise RuntimeError(f"Database initialization failed: {error}")


def test_connection() -> Dict[str, Any]:
    """
    Verifies the active database connection.
    Returns status, dialect, and current database URL info.
    """
    try:
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1")).scalar()
            dialect = engine.dialect.name
            url_display = (
                str(engine.url).replace(engine.url.password or "___", "****")
                if engine.url.password
                else str(engine.url)
            )

            res = {
                "status": "connected" if result == 1 else "unexpected_result",
                "database_type": dialect,
                "is_postgresql": dialect == "postgresql",
                "is_sqlite": dialect == "sqlite",
                "url": url_display,
            }

            if _last_pg_error:
                res["warning"] = "PostgreSQL is offline. Operating in fallback SQLite mode."
                res["postgres_offline_reason"] = _last_pg_error

            return res

    except Exception as error:
        dialect = "unknown"
        try:
            dialect = get_engine().dialect.name
        except Exception:
            pass
        return {
            "status": "error",
            "database_type": dialect,
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
) -> Dict[str, Any]:
    """
    Creates an alert for a suspicious transaction.
    """
    initialize_database()

    if prediction != "Suspicious":
        return {
            "alert_created": False,
            "message": "Transaction does not require an alert.",
        }

    current_time = datetime.now(timezone.utc).isoformat()
    engine = get_engine()

    try:
        stmt = insert(alerts_table).values(
            transaction_id=transaction_id,
            fraud_probability=fraud_probability,
            prediction=prediction,
            risk_score=risk_score,
            risk_level=risk_level,
            alert_status="New",
            created_at=current_time,
            updated_at=current_time,
        )

        with engine.begin() as connection:
            result = connection.execute(stmt)
            alert_id = result.inserted_primary_key[0] if result.inserted_primary_key else None

        return {
            "alert_created": True,
            "alert_id": alert_id,
            "transaction_id": transaction_id,
            "alert_status": "New",
            "message": "Suspicious transaction flagged successfully.",
        }

    except Exception as error:
        raise RuntimeError(f"Failed to create alert: {error}")


# ============================================================
# Get All Alerts
# ============================================================

def get_all_alerts(status: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Returns all alerts.

    Optional:
    status = New / Under Review / Resolved
    """
    initialize_database()
    engine = get_engine()

    try:
        stmt = select(alerts_table).order_by(alerts_table.c.created_at.desc())
        if status:
            stmt = stmt.where(alerts_table.c.alert_status == status)

        with engine.connect() as connection:
            rows = connection.execute(stmt).mappings().all()

        return [dict(row) for row in rows]

    except Exception as error:
        raise RuntimeError(f"Failed to retrieve alerts: {error}")


# ============================================================
# Get Single Alert
# ============================================================

def get_alert(alert_id: int) -> Optional[Dict[str, Any]]:
    """
    Returns one alert using its alert ID.
    """
    initialize_database()
    engine = get_engine()

    try:
        stmt = select(alerts_table).where(alerts_table.c.id == alert_id)

        with engine.connect() as connection:
            row = connection.execute(stmt).mappings().first()

        if row is None:
            return None

        return dict(row)

    except Exception as error:
        raise RuntimeError(f"Failed to retrieve alert: {error}")


# ============================================================
# Update Alert Status
# ============================================================

def update_alert_status(alert_id: int, status: str) -> Dict[str, Any]:
    """
    Updates an alert status.

    Allowed values:
    - New
    - Under Review
    - Resolved
    """
    allowed_statuses = ["New", "Under Review", "Resolved"]

    if status not in allowed_statuses:
        raise ValueError(
            "Invalid alert status. Use: New, Under Review, or Resolved."
        )

    initialize_database()
    engine = get_engine()
    updated_time = datetime.now(timezone.utc).isoformat()

    try:
        stmt = (
            update(alerts_table)
            .where(alerts_table.c.id == alert_id)
            .values(alert_status=status, updated_at=updated_time)
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
            "message": "Alert status updated successfully.",
        }

    except Exception as error:
        raise RuntimeError(f"Failed to update alert status: {error}")


# ============================================================
# Delete Alert
# ============================================================

def delete_alert(alert_id: int) -> Dict[str, Any]:
    """
    Deletes an alert by ID.
    """
    initialize_database()
    engine = get_engine()

    try:
        stmt = delete(alerts_table).where(alerts_table.c.id == alert_id)

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
        raise RuntimeError(f"Failed to delete alert: {error}")


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
                select(func.count()).select_from(alerts_table)
            ).scalar() or 0

            new_alerts = connection.execute(
                select(func.count())
                .select_from(alerts_table)
                .where(alerts_table.c.alert_status == "New")
            ).scalar() or 0

            under_review = connection.execute(
                select(func.count())
                .select_from(alerts_table)
                .where(alerts_table.c.alert_status == "Under Review")
            ).scalar() or 0

            resolved = connection.execute(
                select(func.count())
                .select_from(alerts_table)
                .where(alerts_table.c.alert_status == "Resolved")
            ).scalar() or 0

        return {
            "total_alerts": total_alerts,
            "new_alerts": new_alerts,
            "under_review": under_review,
            "resolved": resolved,
        }

    except Exception as error:
        raise RuntimeError(f"Failed to create alert summary: {error}")


# ============================================================
# SQLite -> PostgreSQL Data Migration Utility
# ============================================================

def migrate_sqlite_to_postgres() -> Dict[str, Any]:
    """
    Migrates existing alerts from the local SQLite database to the
    currently configured PostgreSQL database.
    """
    engine = get_engine()
    if engine.dialect.name != "postgresql":
        return {
            "migrated": False,
            "message": "Target database is not PostgreSQL. Please configure and start PostgreSQL first.",
        }

    if not DATABASE_FILE.exists():
        return {
            "migrated": False,
            "message": f"SQLite database file not found at {DATABASE_FILE}.",
        }

    initialize_database()
    sqlite_engine = sa.create_engine(f"sqlite:///{DATABASE_FILE}")

    try:
        with sqlite_engine.connect() as sqlite_conn:
            rows = sqlite_conn.execute(
                select(
                    alerts_table.c.transaction_id,
                    alerts_table.c.fraud_probability,
                    alerts_table.c.prediction,
                    alerts_table.c.risk_score,
                    alerts_table.c.risk_level,
                    alerts_table.c.alert_status,
                    alerts_table.c.created_at,
                    alerts_table.c.updated_at,
                )
            ).mappings().all()

        if not rows:
            return {"migrated": True, "migrated_count": 0, "message": "No rows to migrate."}

        with engine.begin() as pg_conn:
            for row in rows:
                pg_conn.execute(insert(alerts_table).values(**dict(row)))

        return {
            "migrated": True,
            "migrated_count": len(rows),
            "message": f"Successfully migrated {len(rows)} alerts from SQLite to PostgreSQL.",
        }

    except Exception as error:
        raise RuntimeError(f"Migration failed: {error}")


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":
    try:
        print("Testing Database Connection...")
        conn_info = test_connection()
        print("Connection Info:", conn_info)

        initialize_database()
        print("Alert database initialized successfully.")

        print("\nCurrent alert summary:")
        print(get_alert_summary())

    except Exception as error:
        print(f"\nAlert service test failed:\n{error}")