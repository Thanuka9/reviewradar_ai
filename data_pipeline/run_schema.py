#!/usr/bin/env python3
"""
run_schema.py

Enhances the `users` table by adding extended Yelp columns (if not already present).
Safe to run multiple times — idempotent.

Usage:
    python data_pipeline/run_schema.py
"""

import os
import sys
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

def log(msg): print(msg)

def get_engine():
    load_dotenv()
    user = os.getenv("DB_USER")
    pwd  = os.getenv("DB_PASS")
    host = os.getenv("DB_HOST")
    port = os.getenv("DB_PORT", "5432")
    db   = os.getenv("DB_NAME")

    if not all([user, pwd, host, port, db]):
        log("❌ ERROR: Missing one or more environment variables.")
        sys.exit(1)

    url = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    log(f"🔌 Connecting to {url}")
    return create_engine(url, echo=False)

def apply_user_schema(engine):
    ddl = """
    ALTER TABLE users
      ADD COLUMN IF NOT EXISTS elite               TEXT,
      ADD COLUMN IF NOT EXISTS friends             TEXT,
      ADD COLUMN IF NOT EXISTS fans                INTEGER,
      ADD COLUMN IF NOT EXISTS average_stars       REAL,
      ADD COLUMN IF NOT EXISTS compliment_hot      INTEGER,
      ADD COLUMN IF NOT EXISTS compliment_more     INTEGER,
      ADD COLUMN IF NOT EXISTS compliment_profile  INTEGER,
      ADD COLUMN IF NOT EXISTS compliment_cute     INTEGER,
      ADD COLUMN IF NOT EXISTS compliment_list     INTEGER,
      ADD COLUMN IF NOT EXISTS compliment_note     INTEGER,
      ADD COLUMN IF NOT EXISTS compliment_plain    INTEGER,
      ADD COLUMN IF NOT EXISTS compliment_cool     INTEGER,
      ADD COLUMN IF NOT EXISTS compliment_funny    INTEGER,
      ADD COLUMN IF NOT EXISTS compliment_writer   INTEGER,
      ADD COLUMN IF NOT EXISTS compliment_photos   INTEGER;
    """
    try:
        with engine.begin() as conn:
            conn.execute(text(ddl))
        log("✅ User schema updated successfully.")
        log("📌 Columns added (if missing): elite, friends, fans, average_stars, compliment_*")
    except SQLAlchemyError as e:
        log(f"❌ ERROR: Failed to alter users table — {e}")
        sys.exit(1)

if __name__ == "__main__":
    engine = get_engine()
    apply_user_schema(engine)
