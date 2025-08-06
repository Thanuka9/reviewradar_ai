#!/usr/bin/env python3
"""
inspect_schema.py

Connects to your Postgres via .env and prints out every table's
column names, data types, and row counts. Useful for verifying what
columns (review_features, reviews, etc.) actually exist before
building your pipeline and seeing how many rows each table holds.
"""

import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect, text


def get_engine():
    load_dotenv()  # expects DB_USER, DB_PASS, DB_HOST, DB_PORT, DB_NAME in .env
    return create_engine(
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}"
        f"@{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5432')}"
        f"/{os.getenv('DB_NAME')}"
    )


def main():
    engine = get_engine()
    inspector = inspect(engine)

    print("\n=== Tables in database ===")
    with engine.connect() as conn:
        for table_name in inspector.get_table_names():
            # Print table name
            print(f"\nTable: {table_name}")

            # Print column details
            cols = inspector.get_columns(table_name)
            for col in cols:
                name = col['name']
                dtype = str(col['type'])  # convert to string to avoid format errors
                nullable = col.get('nullable', True)
                default = col.get('default', None)
                print(f"  - {name:<30} {dtype:<20} nullable={nullable} default={default}")

            # Query row count
            try:
                result = conn.execute(text(f"SELECT COUNT(*) AS cnt FROM {table_name}"))
                count = result.scalar()
                print(f"  -> Rows: {count}")
            except Exception as e:
                print(f"  -> Could not get row count: {e}")


if __name__ == "__main__":
    main()
