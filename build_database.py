"""
Loads quality_data.csv into a SQLite database (quality_data.db) for SQL-based
analysis alongside the Python/Dash dashboard and the EDA notebook.

Safe to re-run any number of times: recreates the `quality_data` table from
scratch each time (if_exists="replace"), so it stays in sync with the CSV.
"""

import sqlite3

import pandas as pd

CSV_PATH = "quality_data.csv"
DB_PATH = "quality_data.db"
TABLE_NAME = "quality_data"


def main() -> None:
    df = pd.read_csv(CSV_PATH, parse_dates=["date"])

    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_date ON {TABLE_NAME}(date)"
        )
        conn.execute(
            f"CREATE INDEX IF NOT EXISTS idx_model_category "
            f"ON {TABLE_NAME}(vehicle_model, defect_category)"
        )

    print(f"Loaded {len(df)} rows into {DB_PATH} (table: {TABLE_NAME})")


if __name__ == "__main__":
    main()
