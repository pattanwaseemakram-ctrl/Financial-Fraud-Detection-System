# ============================================================
# Alert and Flagging Service
# Financial Fraud Detection System
# ============================================================

import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


# ============================================================
# Project Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = (
    PROJECT_ROOT
    / "backend"
    / "data"
)

DATABASE_FILE = (
    DATA_DIR
    / "fraud_alerts.db"
)


# ============================================================
# Database Initialization
# ============================================================

def initialize_database():
    """
    Creates the alert database and alerts table if they
    do not already exist.
    """

    try:

        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        with sqlite3.connect(
            DATABASE_FILE
        ) as connection:

            cursor = connection.cursor()

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    transaction_id TEXT NOT NULL,

                    fraud_probability REAL NOT NULL,

                    prediction TEXT NOT NULL,

                    risk_score REAL NOT NULL,

                    risk_level TEXT NOT NULL,

                    alert_status TEXT NOT NULL
                    DEFAULT 'New',

                    created_at TEXT NOT NULL,

                    updated_at TEXT NOT NULL
                )
                """
            )

            connection.commit()

    except sqlite3.Error as error:

        raise RuntimeError(
            f"Database initialization failed: {error}"
        )


# ============================================================
# Create Alert
# ============================================================

def create_alert(
    transaction_id: str,
    fraud_probability: float,
    prediction: str,
    risk_score: float,
    risk_level: str,
):
    """
    Creates an alert for a suspicious transaction.
    """

    initialize_database()

    if prediction != "Suspicious":

        return {
            "alert_created": False,
            "message": "Transaction does not require an alert."
        }

    current_time = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    try:

        with sqlite3.connect(
            DATABASE_FILE
        ) as connection:

            cursor = connection.cursor()

            cursor.execute(
                """
                INSERT INTO alerts (
                    transaction_id,
                    fraud_probability,
                    prediction,
                    risk_score,
                    risk_level,
                    alert_status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    transaction_id,
                    fraud_probability,
                    prediction,
                    risk_score,
                    risk_level,
                    "New",
                    current_time,
                    current_time,
                )
            )

            alert_id = cursor.lastrowid

            connection.commit()

        return {
            "alert_created": True,
            "alert_id": alert_id,
            "transaction_id": transaction_id,
            "alert_status": "New",
            "message": "Suspicious transaction flagged successfully."
        }

    except sqlite3.Error as error:

        raise RuntimeError(
            f"Failed to create alert: {error}"
        )


# ============================================================
# Get All Alerts
# ============================================================

def get_all_alerts(
    status: Optional[str] = None
):
    """
    Returns all alerts.

    Optional:
    status = New / Under Review / Resolved
    """

    initialize_database()

    try:

        with sqlite3.connect(
            DATABASE_FILE
        ) as connection:

            connection.row_factory = sqlite3.Row

            cursor = connection.cursor()

            if status:

                cursor.execute(
                    """
                    SELECT *
                    FROM alerts
                    WHERE alert_status = ?
                    ORDER BY created_at DESC
                    """,
                    (status,)
                )

            else:

                cursor.execute(
                    """
                    SELECT *
                    FROM alerts
                    ORDER BY created_at DESC
                    """
                )

            rows = cursor.fetchall()

        return [
            dict(row)
            for row in rows
        ]

    except sqlite3.Error as error:

        raise RuntimeError(
            f"Failed to retrieve alerts: {error}"
        )


# ============================================================
# Get Single Alert
# ============================================================

def get_alert(
    alert_id: int
):
    """
    Returns one alert using its alert ID.
    """

    initialize_database()

    try:

        with sqlite3.connect(
            DATABASE_FILE
        ) as connection:

            connection.row_factory = sqlite3.Row

            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT *
                FROM alerts
                WHERE id = ?
                """,
                (alert_id,)
            )

            row = cursor.fetchone()

        if row is None:

            return None

        return dict(row)

    except sqlite3.Error as error:

        raise RuntimeError(
            f"Failed to retrieve alert: {error}"
        )


# ============================================================
# Update Alert Status
# ============================================================

def update_alert_status(
    alert_id: int,
    status: str
):
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

    updated_time = (
        datetime.now(
            timezone.utc
        ).isoformat()
    )

    try:

        with sqlite3.connect(
            DATABASE_FILE
        ) as connection:

            cursor = connection.cursor()

            cursor.execute(
                """
                UPDATE alerts

                SET
                    alert_status = ?,
                    updated_at = ?

                WHERE id = ?
                """,
                (
                    status,
                    updated_time,
                    alert_id,
                )
            )

            if cursor.rowcount == 0:

                return {
                    "updated": False,
                    "message": "Alert not found."
                }

            connection.commit()

        return {
            "updated": True,
            "alert_id": alert_id,
            "alert_status": status,
            "message": "Alert status updated successfully."
        }

    except sqlite3.Error as error:

        raise RuntimeError(
            f"Failed to update alert status: {error}"
        )


# ============================================================
# Delete Alert
# ============================================================

def delete_alert(
    alert_id: int
):
    """
    Deletes an alert by ID.
    """

    initialize_database()

    try:

        with sqlite3.connect(
            DATABASE_FILE
        ) as connection:

            cursor = connection.cursor()

            cursor.execute(
                """
                DELETE FROM alerts
                WHERE id = ?
                """,
                (alert_id,)
            )

            if cursor.rowcount == 0:

                return {
                    "deleted": False,
                    "message": "Alert not found."
                }

            connection.commit()

        return {
            "deleted": True,
            "alert_id": alert_id,
            "message": "Alert deleted successfully."
        }

    except sqlite3.Error as error:

        raise RuntimeError(
            f"Failed to delete alert: {error}"
        )


# ============================================================
# Alert Summary
# ============================================================

def get_alert_summary():
    """
    Returns alert counts for dashboard use.
    """

    initialize_database()

    try:

        with sqlite3.connect(
            DATABASE_FILE
        ) as connection:

            cursor = connection.cursor()

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM alerts
                """
            )

            total_alerts = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM alerts
                WHERE alert_status = 'New'
                """
            )

            new_alerts = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM alerts
                WHERE alert_status = 'Under Review'
                """
            )

            under_review = cursor.fetchone()[0]

            cursor.execute(
                """
                SELECT COUNT(*)
                FROM alerts
                WHERE alert_status = 'Resolved'
                """
            )

            resolved = cursor.fetchone()[0]

        return {
            "total_alerts": total_alerts,
            "new_alerts": new_alerts,
            "under_review": under_review,
            "resolved": resolved,
        }

    except sqlite3.Error as error:

        raise RuntimeError(
            f"Failed to create alert summary: {error}"
        )


# ============================================================
# Standalone Test
# ============================================================

if __name__ == "__main__":

    try:

        initialize_database()

        print(
            "Alert database initialized successfully."
        )

        print(
            f"Database location:"
            f"\n{DATABASE_FILE}"
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