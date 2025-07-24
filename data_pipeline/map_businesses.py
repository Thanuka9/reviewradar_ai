#!/usr/bin/env python3
"""
map_businesses.py

Explodes the `categories` column from the businesses table in Postgres
into normalized mapping tables:

  • business_categories — (business_id, category) pairs
  • business_geo        — (business_id, latitude, longitude)

Usage:
    python data_pipeline/map_businesses.py
"""

import os
import sys
import pandas as pd
from sqlalchemy import create_engine, String, Float, text
from sqlalchemy.exc import ProgrammingError
from dotenv import load_dotenv

VERBOSE = True

def log(msg):
    if VERBOSE:
        print(msg)

def get_engine():
    load_dotenv()
    user = os.getenv("DB_USER")
    pwd  = os.getenv("DB_PASS")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    db   = os.getenv("DB_NAME")

    missing = [k for k, v in {
        "DB_USER": user, "DB_PASS": pwd, "DB_HOST": host, "DB_PORT": port, "DB_NAME": db
    }.items() if not v]

    if missing:
        print(f"❌ ERROR: Missing env vars: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    url = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    log(f"🔌 Connecting to {url}")
    return create_engine(url, echo=False, future=True)

def drop_derived_tables(engine):
    with engine.begin() as conn:
        conn.execute(text("""
            DROP TABLE IF EXISTS 
                business_categories, 
                business_geo, 
                review_features, 
                user_features, 
                business_features 
            CASCADE;
        """))
    log("🧨 Old derived tables dropped: categories, geo, features")

def main():
    engine = get_engine()
    drop_derived_tables(engine)

    try:
        log("1️⃣  Loading businesses from database...")
        df = pd.read_sql(
            "SELECT business_id, latitude, longitude, categories FROM businesses",
            engine
        )
    except ProgrammingError as e:
        print("❌ ERROR: Could not read `businesses` table. Did you run scraper_businesses.py?")
        sys.exit(1)

    log(f"   → Loaded {len(df)} businesses")

    # Explode comma-separated categories
    log("2️⃣  Exploding categories into (business, category)...")
    df_clean = df.dropna(subset=["categories"]).copy()
    df_clean["category"] = df_clean["categories"].str.split(", ")

    df_cat = (
        df_clean
        .explode("category")
        [["business_id", "category"]]
        .drop_duplicates()
    )
    log(f"   → {len(df_cat)} category mappings created")

    # Geo data
    df_geo = df[["business_id", "latitude", "longitude"]].drop_duplicates()
    log(f"3️⃣  Extracted geo data for {len(df_geo)} businesses")

    # Write to DB
    log("4️⃣  Writing business_categories to DB...")
    df_cat.to_sql(
        "business_categories",
        engine,
        if_exists="replace",
        index=False,
        dtype={
            "business_id": String,
            "category": String,
        }
    )
    log("   ✅ business_categories written")

    log("5️⃣  Writing business_geo to DB...")
    df_geo.to_sql(
        "business_geo",
        engine,
        if_exists="replace",
        index=False,
        dtype={
            "business_id": String,
            "latitude": Float,
            "longitude": Float,
        }
    )
    log("   ✅ business_geo written")

    log("✅ map_businesses.py complete.")

if __name__ == "__main__":
    main()
