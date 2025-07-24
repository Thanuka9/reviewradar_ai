#!/usr/bin/env python3
"""
scraper_users_reviews.py

Idempotent, schema-aware loader for Yelp raw JSON:
- Applies DB migrations if present
- Drops & (re)creates users & reviews tables only
- Dynamically filters JSON fields to match schema (including migrated columns)
- Batches inserts with progress logs
- Loads users first, then reviews (so FK checks pass)
- Logs all operations to console & file
"""
import os
import json
import sys
import logging
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# Load environment
load_dotenv()

# --- Logging Setup ---
LOG_DIR = 'logs'
LOG_PATH = os.path.join(LOG_DIR, 'scraper_users_reviews.log')
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger('scraper_users_reviews')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH, encoding='utf-8')
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_engine():
    """Return SQLAlchemy engine or exit on failure."""
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/"
        f"{os.getenv('DB_NAME')}"
    )
    try:
        eng = create_engine(url, echo=False)
        logger.info(f"Connected to DB: {os.getenv('DB_NAME')}@{os.getenv('DB_HOST')}")
        return eng
    except SQLAlchemyError as e:
        logger.error(f"DB connection failed: {e}")
        sys.exit(1)


def apply_migrations(engine):
    """Run all .sql migrations in migrations/ (if any)."""
    mig_dir = 'migrations'
    if not os.path.isdir(mig_dir):
        logger.info("No migrations directory, skipping migrations.")
        return
    for fname in sorted(os.listdir(mig_dir)):
        if fname.endswith('.sql'):
            path = os.path.join(mig_dir, fname)
            sql = open(path, 'r', encoding='utf-8').read()
            try:
                with engine.begin() as conn:
                    conn.execute(text(sql))
                logger.info(f"Applied migration: {fname}")
            except SQLAlchemyError as e:
                logger.error(f"Migration {fname} failed: {e}")
                sys.exit(1)


def drop_users_reviews(engine):
    """Drop only users & reviews tables to preserve businesses."""
    sql = "DROP TABLE IF EXISTS reviews, users CASCADE;"
    with engine.begin() as conn:
        conn.execute(text(sql))
    logger.info("Dropped users & reviews tables (if existed)")


def create_users_reviews_tables(engine):
    """Create base users & reviews tables."""
    ddl = """
    CREATE TABLE IF NOT EXISTS users (
        user_id TEXT PRIMARY KEY,
        name TEXT,
        review_count INTEGER,
        yelping_since DATE,
        useful INTEGER,
        funny INTEGER,
        cool INTEGER
    );
    CREATE TABLE IF NOT EXISTS reviews (
        review_id TEXT PRIMARY KEY,
        user_id TEXT REFERENCES users(user_id),
        business_id TEXT REFERENCES businesses(business_id),
        stars FLOAT,
        date DATE,
        text TEXT,
        useful INTEGER,
        funny INTEGER,
        cool INTEGER
    );
    """
    with engine.begin() as conn:
        conn.execute(text(ddl))
    logger.info("Created users & reviews tables (if missing)")


def clean_record(data, table):
    """Normalize JSON fields per table."""
    if table == 'users':
        data['yelping_since'] = data.get('yelping_since', '')[:10]
    elif table == 'reviews':
        data['date'] = data.get('date', '')[:10]
    return data


def insert_stmt(table, cols):
    """Generate parameterized INSERT statement."""
    cols_sql = ", ".join(cols)
    vals_sql = ", ".join(f":{c}" for c in cols)
    return text(
        f"INSERT INTO {table}({cols_sql}) VALUES ({vals_sql}) "
        "ON CONFLICT DO NOTHING;"
    )


def load_json(filepath, table, engine, batch_size=1000):
    """
    Load a JSON-lines file into `table` in batches,
    dynamically filtering to existing columns and skipping orphan reviews.
    """
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        logger.warning(f"Table '{table}' missing, skipping load.")
        return

    cols = [c['name'] for c in inspector.get_columns(table)]
    valid_users = valid_biz = None
    if table == 'reviews':
        with engine.begin() as conn:
            valid_users = {r[0] for r in conn.execute(text("SELECT user_id FROM users"))}
            valid_biz = {r[0] for r in conn.execute(text("SELECT business_id FROM businesses"))}

    stmt = None
    batch, total = [], 0
    logger.info(f"Loading '{table}' from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as fh:
            for line in fh:
                rec = clean_record(json.loads(line), table)
                # keep only columns that exist in the table
                rec = {k: v for k, v in rec.items() if k in cols}
                # skip orphaned reviews
                if table == 'reviews' and (
                   rec.get('user_id') not in valid_users or
                   rec.get('business_id') not in valid_biz):
                    continue
                if stmt is None:
                    stmt = insert_stmt(table, list(rec.keys()))
                batch.append(rec)
                total += 1
                if len(batch) >= batch_size:
                    with engine.begin() as conn:
                        conn.execute(stmt, batch)
                    logger.info(f"{table}: batch committed ({total} rows)")
                    batch.clear()

        if batch:
            with engine.begin() as conn:
                conn.execute(stmt, batch)
            logger.info(f"{table}: final batch committed ({total} rows)")

        logger.info(f"Finished loading '{table}': {total} rows")
    except Exception as e:
        logger.error(f"Error loading '{table}': {e}")
        sys.exit(1)


if __name__ == '__main__':
    engine = get_engine()

    # 1) Drop & recreate users+reviews (keep businesses intact)
    drop_users_reviews(engine)
    create_users_reviews_tables(engine)

    # 2) Apply any ALTER TABLE migrations so extended columns are present
    apply_migrations(engine)

    # 3) Load users first, then reviews
    load_json('data/raw/yelp_academic_dataset_user.json',   'users',   engine)
    load_json('data/raw/yelp_academic_dataset_review.json', 'reviews', engine)

    logger.info("✔️  All users & reviews successfully loaded.")
