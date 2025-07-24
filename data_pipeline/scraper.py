#!/usr/bin/env python3
"""
scraper.py

Idempotent, schema-aware loader for Yelp raw JSON
- Drops & (re)creates raw tables
- Applies DB migrations if present
- Dynamically filters JSON fields to match schema
- Batches inserts with progress logs
- Loads businesses & users first, then reviews
- Logs all operations to console & file
"""
import os
import json
import sys
import logging
import argparse
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed

# Ensure console can print UTF-8
if hasattr(sys.stdout, "reconfigure"):  # Python 3.7+
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment
load_dotenv()

# --- Logging Setup ---
LOG_DIR = 'logs'
LOG_PATH = os.path.join(LOG_DIR, 'scraper.log')
os.makedirs(LOG_DIR, exist_ok=True)
logger = logging.getLogger('production_scraper')
logger.setLevel(logging.INFO)
fh = logging.FileHandler(LOG_PATH, encoding='utf-8')
formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")
fh.setFormatter(formatter)
logger.addHandler(fh)
ch = logging.StreamHandler(sys.stdout)
ch.setFormatter(formatter)
logger.addHandler(ch)


def get_engine():
    """Builds and returns a SQLAlchemy engine, exits on failure."""
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
    """Executes any .sql files in the migrations/ directory in sorted order."""
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


def drop_raw_tables(engine):
    """Drops raw tables if they exist."""
    sql = "DROP TABLE IF EXISTS reviews, users, businesses CASCADE;"
    with engine.begin() as conn:
        conn.execute(text(sql))
    logger.info("Dropped raw tables (if existed)")


def create_raw_tables(engine):
    """Creates businesses, users, reviews tables with base schema."""
    ddl = """
    CREATE TABLE IF NOT EXISTS businesses (
        business_id TEXT PRIMARY KEY,
        name TEXT,
        address TEXT,
        city TEXT,
        state TEXT,
        postal_code TEXT,
        latitude FLOAT,
        longitude FLOAT,
        stars FLOAT,
        review_count INTEGER,
        is_open BOOLEAN,
        attributes JSONB,
        categories TEXT,
        hours JSONB
    );

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
    logger.info("Created raw tables (if missing)")


def clean_record(data, table):
    """Normalize JSON fields per table."""
    if table == 'businesses':
        data['is_open'] = bool(data.get('is_open'))
        if isinstance(data.get('attributes'), dict):
            data['attributes'] = json.dumps(data['attributes'])
        if isinstance(data.get('hours'), dict):
            data['hours'] = json.dumps(data['hours'])
    elif table == 'users':
        data['yelping_since'] = data.get('yelping_since','')[:10]
    elif table == 'reviews':
        data['date'] = data.get('date','')[:10]
    return data


def insert_stmt(table, cols):
    """Generates INSERT statement for given table and columns."""
    cols_sql = ", ".join(cols)
    vals_sql = ", ".join(f":{c}" for c in cols)
    return text(f"INSERT INTO {table}({cols_sql}) VALUES ({vals_sql}) ON CONFLICT DO NOTHING;")


def load_json(filepath, table, engine, batch_size):
    """
    Loads a JSON-lines file into the given table in batches,
    filtering only existing columns and skipping orphaned reviews.
    """
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        logger.warning(f"Table '{table}' missing, skipping load.")
        return

    table_cols = [col['name'] for col in inspector.get_columns(table)]
    valid_users = valid_biz = None
    if table == 'reviews':
        with engine.begin() as conn:
            valid_users = {r[0] for r in conn.execute(text("SELECT user_id FROM users"))}
            valid_biz = {r[0] for r in conn.execute(text("SELECT business_id FROM businesses"))}

    stmt = None
    batch, total = [], 0
    logger.info(f"Loading '{table}' from {filepath}...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                rec = clean_record(json.loads(line), table)
                # keep only columns that exist on the table
                rec = {k: rec[k] for k in table_cols if k in rec}
                if table == 'reviews':
                    if rec.get('user_id') not in valid_users or rec.get('business_id') not in valid_biz:
                        continue
                if stmt is None:
                    # stable column ordering
                    cols = list(rec.keys())
                    stmt = insert_stmt(table, cols)
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


def main(batch_size):
    engine = get_engine()

    # build schema
    drop_raw_tables(engine)
    create_raw_tables(engine)
    apply_migrations(engine)

    # load businesses & users first
    datasets = [
        ('businesses', 'data/raw/yelp_academic_dataset_business.json'),
        ('users',      'data/raw/yelp_academic_dataset_user.json'),
    ]
    with ThreadPoolExecutor(max_workers=2) as exec:
        futures = {exec.submit(load_json, path, tbl, engine, batch_size): tbl for tbl, path in datasets}
        for fut in as_completed(futures):
            tbl = futures[fut]
            try:
                fut.result()
            except Exception as e:
                logger.error(f"{tbl} failed: {e}")
                sys.exit(1)

    # then load reviews
    load_json('data/raw/yelp_academic_dataset_review.json', 'reviews', engine, batch_size)

    logger.info("All raw tables successfully loaded.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load Yelp raw JSON into Postgres')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of rows per insert batch')
    args = parser.parse_args()
    main(args.batch_size)
