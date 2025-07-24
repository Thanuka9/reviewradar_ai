#!/usr/bin/env python3
"""
features.py

Builds three feature tables with extended features and records them to disk.
If the normal chunked to_sql fails (e.g. OOM), falls back to CSV + COPY.

  • review_features   — per-review
  • user_features     — per-user aggregates
  • business_features — per-business aggregates

All stdout is logged to outputs/features.log; all plots are saved under outputs/.
Numeric features are scaled (StandardScaler → outputs/scaler.pkl).
Usage:
    python data_pipeline/features.py [--live]
"""
import os
import sys
import time
import math
import pickle
import tempfile
import argparse
from datetime import datetime
import re

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from pandas.tseries.holiday import USFederalHolidayCalendar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Buffer all log lines for final write
_LOG_LINES = []

def log_print(msg: str):
    """Print + buffer for logfile."""
    line = f"{datetime.now().isoformat()}  {msg}"
    print(line)
    _LOG_LINES.append(line)

def get_engine():
    """Connect to Postgres via env vars."""
    load_dotenv()
    url = (
        f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
        f"{os.getenv('DB_HOST','localhost')}:{os.getenv('DB_PORT','5432')}/"
        f"{os.getenv('DB_NAME')}"
    )
    try:
        engine = create_engine(url, echo=False, future=True)
        # smoke-test
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        log_print(f"🔌 Connected to DB at {url}")
        return engine
    except Exception as e:
        log_print(f"❌ DB connection failed: {e}")
        sys.exit(1)

def save_plot(fig, name):
    """Save a matplotlib figure into outputs/ and log."""
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", name)
    fig.savefig(path, bbox_inches="tight")
    log_print(f"   • Plot saved to {path}")

def safe_sql_read(query, engine, label, parse_dates=None):
    """Read SQL into DataFrame, catching errors."""
    try:
        return pd.read_sql(query, engine, parse_dates=parse_dates)
    except Exception as e:
        log_print(f"❌ Failed to read {label}: {e}")
        return pd.DataFrame()

def copy_from_csv(df: pd.DataFrame, cols: list, engine, table: str):
    """Fallback bulk load via CSV + COPY."""
    fd, tmp_path = tempfile.mkstemp(suffix=".csv")
    os.close(fd)
    log_print("   • Falling back: writing temp CSV for COPY")
    df.to_csv(tmp_path, columns=cols, index=False)
    log_print(f"   • Temp CSV written to {tmp_path}, invoking COPY")
    raw = engine.raw_connection()
    cur = raw.cursor()
    try:
        with open(tmp_path, "r", encoding="utf-8") as f:
            cur.copy_expert(
                f'COPY {table}({",".join(cols)}) FROM STDIN WITH CSV HEADER',
                f
            )
        raw.commit()
        log_print(f"   ✓ COPY loaded {len(df)} rows into {table}")
    except Exception as e:
        raw.rollback()
        log_print(f"❌ COPY fallback failed: {e}")
        sys.exit(1)
    finally:
        cur.close()
        raw.close()
        os.remove(tmp_path)

def show_live_plot():
    """Demo live updating plot."""
    import random
    fig, ax = plt.subplots()
    xs, ys = [], []
    def animate(i):
        xs.append(i)
        ys.append(random.randint(0,100))
        ax.clear()
        ax.plot(xs, ys)
        ax.set_title("Live Data Plot")
    animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()

def main(show_live=False):
    os.makedirs("outputs", exist_ok=True)
    engine   = get_engine()
    analyzer = SentimentIntensityAnalyzer()
    holidays = USFederalHolidayCalendar().holidays()

    # 1) Load reviews
    log_print("1) Loading raw reviews…")
    df = safe_sql_read(
        "SELECT review_id, user_id, r.business_id, stars, text, date::timestamptz AS date "
        "FROM reviews r",
        engine, "reviews", parse_dates=["date"]
    )
    if df.empty:
        log_print("❌ No reviews found. Exiting.")
        return
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    log_print(f"   → {len(df)} reviews loaded")

    # 2) Sentiment & length
    log_print("   • computing VADER sentiment & text length")
    df["sentiment"]   = df["text"].map(lambda t: analyzer.polarity_scores(str(t))["compound"])
    df["text_length"] = df["text"].str.len().fillna(0).astype(int)
    save_plot(df["sentiment"].hist(bins=50).get_figure(), "sentiment_hist.png")
    save_plot(df["text_length"].hist(bins=50).get_figure(), "text_length_hist.png")

    # 3) TF-IDF
    log_print("   • computing TF-IDF (top 50 terms)")
    tfidf = TfidfVectorizer(max_features=50, stop_words="english")
    X     = tfidf.fit_transform(df["text"].fillna(""))
    tfidf_cols = [f"tfidf_{c}" for c in tfidf.get_feature_names_out()]
    df_tfidf   = pd.DataFrame(X.toarray(), columns=tfidf_cols, index=df.index).round(5)
    df = pd.concat([df, df_tfidf], axis=1)
    save_plot(df_tfidf.mean().sort_values().plot.barh(figsize=(6,8)).get_figure(), "tfidf_top50.png")

    # 4) Keyword counts
    log_print("   • counting keywords")
    for kw in ["delivery","service","price"]:
        df[f"kw_{kw}"] = (
            df["text"].str.count(rf"\b{kw}\b", flags=re.IGNORECASE)
              .fillna(0).astype(int)
        )
    save_plot(df[[f"kw_{k}" for k in ["delivery","service","price"]]]
              .sum().plot.bar().get_figure(), "keyword_counts.png")

    # 5) Temporal features
    log_print("   • extracting temporal features")
    df["dow"]            = df["date"].dt.dayofweek
    df["hour"]           = df["date"].dt.hour
    df["is_weekend"]     = df["dow"].isin([5,6]).astype(int)
    df["is_holiday"]     = df["date"].dt.normalize().isin(holidays).astype(int)
    df["month"]          = df["date"].dt.month
    df["quarter"]        = df["date"].dt.quarter
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"]   = df["date"].dt.is_month_end.astype(int)
    df["weekofyear"]     = df["date"].dt.isocalendar().week
    save_plot(df["dow"].value_counts().sort_index().plot.bar().get_figure(), "dow_distribution.png")
    save_plot(df["hour"].value_counts().sort_index().plot.bar().get_figure(), "hour_distribution.png")

    # 6) Geo / category_count / state join
    log_print("   • joining geo, category_count and state")
    biz_geo   = safe_sql_read("SELECT business_id, latitude, longitude FROM business_geo", engine, "business_geo")
    biz_cat   = safe_sql_read("SELECT business_id, COUNT(*) AS category_count FROM business_categories GROUP BY business_id", engine, "business_categories")
    biz_state = safe_sql_read("SELECT business_id, state FROM businesses", engine, "businesses")
    df = (
        df.merge(biz_geo, on="business_id", how="left")
          .merge(biz_cat, on="business_id", how="left")
          .merge(biz_state, on="business_id", how="left")
    )
    df["lat_round"] = df["latitude"].round(2)
    df["lon_round"] = df["longitude"].round(2)

    # 7) Geo clustering
    log_print("   • fitting KMeans (20 clusters)")
    gc = biz_geo.dropna(subset=["latitude","longitude"]).reset_index(drop=True)
    km = KMeans(n_clusters=20, random_state=42).fit(gc[["latitude","longitude"]])
    clust = gc[["business_id"]].copy()
    clust["geo_cluster"] = km.labels_
    df = df.merge(clust, on="business_id", how="left")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.scatter(gc["longitude"], gc["latitude"], c=km.labels_, cmap="tab20", s=5, alpha=0.6)
    ax.set(title="Geo Clusters", xlabel="Longitude", ylabel="Latitude")
    save_plot(fig, "kmeans_clusters.png")

    # 8) Rolling user avg ratings
    log_print("   • computing rolling user avg ratings")
    df = df.dropna(subset=["date"]).sort_values(["user_id","date"]).set_index("date")
    for window,label in [(90,"3m"),(180,"6m"),(365,"12m")]:
        col = f"user_avg_{label}"
        df[col] = df.groupby("user_id")["stars"].rolling(f"{window}D").mean().values
        save_plot(df[col].hist(bins=50).get_figure(), f"rolling_{label}_hist.png")
    df = df.reset_index()

    # 9) One-hot top 10 categories
    log_print("   • one-hot encoding top 10 categories")
    top10 = safe_sql_read(
        "SELECT category FROM business_categories GROUP BY category ORDER BY COUNT(*) DESC LIMIT 10",
        engine, "top categories"
    )["category"].tolist()
    bc     = safe_sql_read("SELECT business_id, category FROM business_categories", engine, "business_categories")
    bc_top = bc[bc["category"].isin(top10)]
    ohe    = pd.get_dummies(bc_top["category"], prefix="cat")
    ohe["business_id"] = bc_top["business_id"]
    ohe = ohe.groupby("business_id").max().reset_index()
    df  = df.merge(ohe, on="business_id", how="left")\
            .fillna({c:0 for c in ohe.columns if c.startswith("cat_")})
    save_plot(ohe.drop(columns="business_id").sum().sort_values().plot.barh().get_figure(),
              "cat_onehot_counts.png")

    # 10) Write review_features (streaming in chunks with ETA)
    log_print("   • writing review_features to DB (streaming in chunks)")
    cols = (
        ["review_id","user_id","business_id","stars","sentiment","text_length",
         "dow","hour","is_weekend","is_holiday","month","quarter","is_month_start",
         "is_month_end","weekofyear","latitude","longitude","lat_round","lon_round",
         "geo_cluster","category_count","state"]
        + tfidf_cols
        + [f"kw_{k}" for k in ["delivery","service","price"]]
        + [f"user_avg_{l}" for _,l in [(90,"3m"),(180,"6m"),(365,"12m")]]
        + [c for c in ohe.columns if c.startswith("cat_")]
    )
    n = len(df)
    chunksize   = 10_000
    num_chunks  = math.ceil(n / chunksize)
    start_time  = time.time()

    # drop via transaction
    with engine.begin() as conn:
        conn.execute(text("DROP TABLE IF EXISTS review_features"))

    first = True
    for idx, offset in enumerate(range(0, n, chunksize), 1):
        chunk = df.iloc[offset:offset+chunksize][cols]
        try:
            chunk.to_sql(
                "review_features",
                engine,
                if_exists="append" if not first else "replace",
                index=False,
                method="multi"
            )
        except Exception as e:
            log_print(f"❌ Chunk {idx} failed: {e}; falling back to CSV+COPY for entire table")
            copy_from_csv(df, cols, engine, "review_features")
            break

        elapsed = time.time() - start_time
        avg     = elapsed / idx
        rem     = num_chunks - idx
        eta     = avg * rem
        log_print(f"   • chunk {idx}/{num_chunks} ({len(chunk)} rows) – "
                  f"{elapsed:.1f}s elapsed, ETA {eta:.1f}s")
        first = False

    # 11) user_features
    log_print("2) Building user_features…")
    user_agg = df.groupby("user_id").agg(
        avg_sentiment=("sentiment","mean"),
        total_reviews=("review_id","count"),
        avg_length=("text_length","mean")
    ).reset_index()
    user_agg.to_sql("user_features", engine, if_exists="replace", index=False)
    log_print(f"   ✓ Wrote {len(user_agg)} rows to user_features")

    # 12) business_features
    log_print("3) Building business_features…")
    biz_agg = df.groupby("business_id").agg(
        avg_stars=("stars","mean"),
        avg_sentiment=("sentiment","mean"),
        total_reviews=("review_id","count")
    ).reset_index()
    biz_agg.to_sql("business_features", engine, if_exists="replace", index=False)
    log_print(f"   ✓ Wrote {len(biz_agg)} rows to business_features")

    # 13) Scale numeric
    log_print("4) Scaling numeric features")
    num_cols = [c for c in cols if c not in ("review_id","user_id","business_id","state")]
    scaler   = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols].fillna(0))
    with open("outputs/scaler.pkl","wb") as f:
        pickle.dump(scaler, f)
    log_print("   ✓ Saved StandardScaler to outputs/scaler.pkl")

    # Finish
    log_print("✅ features.py complete — all feature tables and plots are up to date.")
    with open("outputs/features.log","w", encoding="utf-8") as f:
        f.write("\n".join(_LOG_LINES))
    print("📝 Run log written to outputs/features.log")

    if show_live:
        show_live_plot()

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--live", action="store_true", help="Show live plot at end")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main(show_live=args.live)
