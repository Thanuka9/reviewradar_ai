#!/usr/bin/env python3
"""
Enhanced Standalone EDA Script for Yelp Dataset
This script:
  - Connects to the Postgres database
  - Loads a sample of reviews, business categories, and user join dates
  - Computes & plots (saving all to outputs/):
      1) Star rating distribution
      2) Review length distribution
      3) Top 20 business categories
      4) Users by join year
      5) Monthly review volume
      6) Average rating by category
      7) Reviews per business distribution
      8) Review length vs. star rating boxplot
      9) Top 20 most active users
     10) Sentiment vs. rating
     11) Time-series decomposition of monthly volume (if ≥24 months)
     12) User retention (lifetime)
  - Generates a Markdown report `outputs/eda_report.md`

Usage:
    python notebooks/eda.py

Requirements:
    pandas, sqlalchemy, python-dotenv, matplotlib, psycopg2, tabulate, statsmodels, vaderSentiment
"""

import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from datetime import datetime
import platform, pkg_resources

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from statsmodels.tsa.seasonal import seasonal_decompose

# Optional dependency for markdown tables
try:
    import tabulate  # noqa: F401
except ImportError:
    tabulate = None
    print("Warning: 'tabulate' not installed. Markdown tables will fall back to plain text.")


def to_markdown_or_fallback(obj):
    try:
        return obj.to_markdown()
    except Exception:
        return obj.to_string()


def get_engine():
    load_dotenv()
    user = os.getenv("DB_USER")
    pwd  = os.getenv("DB_PASS")
    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT") or "5432"
    db   = os.getenv("DB_NAME")
    if not all([user, pwd, host, port, db]):
        raise RuntimeError(
            f"Missing DB env var: DB_USER={user!r}, DB_PASS={'set' if pwd else 'missing'}, "
            f"DB_HOST={host!r}, DB_PORT={port!r}, DB_NAME={db!r}"
        )
    url = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    print(f"🔌 Connecting to {url}")
    return create_engine(url, echo=False)


def save_plot(fig, name):
    os.makedirs("outputs", exist_ok=True)
    path = os.path.join("outputs", name)
    fig.savefig(path, bbox_inches="tight")
    print(f"  • Plot saved to {path}")
    return path


def main():
    engine = get_engine()
    # record versions
    ts  = datetime.now().isoformat()
    pyv = platform.python_version()
    pdv = pkg_resources.get_distribution("pandas").version
    sav = pkg_resources.get_distribution("SQLAlchemy").version
    mpv = pkg_resources.get_distribution("matplotlib").version
    svs = pkg_resources.get_distribution("statsmodels").version

    report = [
        "# EDA Report for Yelp Dataset",
        f"**Run on:** {ts}",
        (
            f"**Environment:** Python {pyv}, pandas {pdv}, "
            f"SQLAlchemy {sav}, matplotlib {mpv}, statsmodels {svs}"
        ),
        "---\n",
    ]

    # 1) Reviews sample
    report.append("## 1) Reviews Sample")
    df_reviews = pd.read_sql(
        "SELECT review_id, user_id, business_id, stars, text, date::date AS date FROM reviews LIMIT 100000",
        engine,
        parse_dates=["date"],
    )
    report.append(f"- Loaded **{len(df_reviews)}** reviews (sample)\n")

    # 2) Star rating distribution
    report.append("## 2) Star Rating Distribution")
    star_counts = df_reviews["stars"].value_counts().sort_index()
    report.append(to_markdown_or_fallback(star_counts))
    fig = star_counts.plot.bar().get_figure()
    save_plot(fig, "star_distribution.png"); plt.clf()
    report.append("![Stars](outputs/star_distribution.png)\n")

    # 3) Review length distribution
    report.append("## 3) Review Length Distribution")
    df_reviews["text_length"] = df_reviews["text"].str.len()
    report.append(to_markdown_or_fallback(df_reviews["text_length"].describe()))
    fig = df_reviews["text_length"].hist(bins=50).get_figure()
    save_plot(fig, "review_length.png"); plt.clf()
    report.append("![Length](outputs/review_length.png)\n")

    # 4) Top 20 business categories
    report.append("## 4) Top 20 Business Categories")
    df_biz = pd.read_sql("SELECT categories FROM businesses", engine)
    cats = df_biz["categories"].dropna().str.split(", ").explode()
    top_cats = cats.value_counts().head(20)
    report.append(to_markdown_or_fallback(top_cats))
    fig = top_cats.plot.barh().get_figure()
    save_plot(fig, "top_categories.png"); plt.clf()
    report.append("![Categories](outputs/top_categories.png)\n")

    # 5) Users by join year
    report.append("## 5) Users by Join Year")
    df_users = pd.read_sql("SELECT yelping_since::date AS since FROM users", engine, parse_dates=["since"])
    years = df_users["since"].dt.year.dropna().astype(int)
    yc = years.value_counts().sort_index()
    report.append(to_markdown_or_fallback(yc))
    fig = yc.plot().get_figure()
    save_plot(fig, "users_by_join_year.png"); plt.clf()
    report.append("![Join Year](outputs/users_by_join_year.png)\n")

    # 6) Monthly review volume
    report.append("## 6) Monthly Review Volume")
    monthly = pd.read_sql(
        "SELECT date_trunc('month', date::date) AS month, COUNT(*) AS cnt "
        "FROM reviews GROUP BY month ORDER BY month",
        engine,
        parse_dates=["month"],
    ).dropna(subset=["month"])
    # strip any tz and coerce to datetime64[ns]
    monthly["month"] = pd.to_datetime(monthly["month"].dt.tz_localize(None))
    report.append(to_markdown_or_fallback(monthly.set_index("month")))
    fig = monthly.set_index("month")["cnt"].plot().get_figure()
    save_plot(fig, "monthly_volume.png"); plt.clf()
    report.append("![Monthly](outputs/monthly_volume.png)\n")

    # 7) Average rating by category
    report.append("## 7) Average Rating by Category (Top 20)")
    df_join = pd.read_sql(
        "SELECT r.stars, b.categories FROM reviews r "
        "JOIN businesses b ON r.business_id=b.business_id LIMIT 100000",
        engine,
    )
    df_join["cat"] = df_join["categories"].str.split(", ")
    ex = df_join.explode("cat")
    avg_cat = ex.groupby("cat")["stars"].mean().sort_values(ascending=False).head(20)
    report.append(to_markdown_or_fallback(avg_cat))
    fig = avg_cat.plot.barh().get_figure()
    save_plot(fig, "avg_rating_by_category.png"); plt.clf()
    report.append("![AvgRating](outputs/avg_rating_by_category.png)\n")

    # 8) Reviews per business distribution
    report.append("## 8) Reviews per Business Distribution")
    rc = pd.read_sql(
        "SELECT business_id, COUNT(*) AS cnt FROM reviews GROUP BY business_id", engine
    )["cnt"]
    report.append(to_markdown_or_fallback(rc.describe()))
    fig = rc.hist(bins=50).get_figure()
    save_plot(fig, "reviews_per_business.png"); plt.clf()
    report.append("![BizDist](outputs/reviews_per_business.png)\n")

    # 9) Review length vs. star rating boxplot
    report.append("## 9) Review Length vs. Star Rating")
    fig = df_reviews.boxplot(column="text_length", by="stars").get_figure()
    save_plot(fig, "length_by_star.png"); plt.clf()
    report.append("![LenVsStar](outputs/length_by_star.png)\n")

    # 10) Top 20 most active users
    report.append("## 10) Top 20 Most Active Users")
    tu = pd.read_sql(
        "SELECT user_id, review_count FROM users ORDER BY review_count DESC LIMIT 20", engine
    ).set_index("user_id")
    report.append(to_markdown_or_fallback(tu))
    fig = tu["review_count"].plot.barh().get_figure()
    save_plot(fig, "top_users.png"); plt.clf()
    report.append("![TopUsers](outputs/top_users.png)\n")

    # 11) Sentiment vs. star rating
    report.append("## 11) Sentiment vs. Star Rating")
    analyzer = SentimentIntensityAnalyzer()
    df_reviews["sentiment"] = df_reviews["text"].apply(
        lambda t: analyzer.polarity_scores(str(t))["compound"]
    )
    corr = df_reviews[["stars", "sentiment"]].corr().iloc[0, 1]
    report.append(f"- Pearson correlation (stars, sentiment) = {corr:.3f}")
    fig = df_reviews.boxplot(column="sentiment", by="stars").get_figure()
    save_plot(fig, "sentiment_by_star.png"); plt.clf()
    report.append("![Sentiment](outputs/sentiment_by_star.png)\n")

    # 12) Seasonal decomposition if enough data
    report.append("## 12) Seasonal Decomposition (if ≥24 months)")
    if len(monthly) >= 24:
        decomp = seasonal_decompose(
            monthly.set_index("month")["cnt"], model="additive", period=12, extrapolate_trend="freq"
        )
        fig = decomp.plot().get_figure(); plt.suptitle("Seasonal Decompose")
        save_plot(fig, "ts_decomposition.png"); plt.clf()
        report.append("![Decomp](outputs/ts_decomposition.png)\n")
    else:
        report.append(f"- Skipped: only {len(monthly)} months of data available\n")

    # 13) User retention (lifetime in days)
    report.append("## 13) User Retention (Lifetime in Days)")
    df_dates = pd.read_sql(
        "SELECT user_id, date::date AS d FROM reviews", engine, parse_dates=["d"]
    )
    grp = df_dates.groupby("user_id")["d"]
    lifetimes = (grp.max() - grp.min()).dt.days
    report.append(to_markdown_or_fallback(lifetimes.describe()))
    fig = lifetimes.hist(bins=50).get_figure()
    save_plot(fig, "user_retention.png"); plt.clf()
    report.append("![Retention](outputs/user_retention.png)\n")

    # Write final Markdown report
    os.makedirs("outputs", exist_ok=True)
    md_path = os.path.join("outputs", "eda_report.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    print(f"✨ EDA report written to {md_path}")


if __name__ == "__main__":
    main()
