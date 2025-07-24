#!/usr/bin/env python3
# run_pipeline.py

import subprocess
import datetime
import os
import sys

# === Config ===
LOG_DIR   = "logs"
LOG_PATH  = os.path.join(LOG_DIR, "pipeline.log")
USE_LOG_FILE = True  # Set False to disable log file output

# === Ordered Pipeline Steps ===
STEPS = [
    ("🔁 Reset & Load Raw Tables",              "python data_pipeline/scraper.py"),
    ("🧱 Apply DB Migrations & Schema Enhancements", "python data_pipeline/run_schema.py"),
    ("📥 Reload Users & Reviews with Extended Fields", "python data_pipeline/scraper_users_reviews.py"),
    ("🗺️  Map Business Categories & Geo",         "python data_pipeline/map_businesses.py"),
    ("🧠 Generate Features & Save Outputs",       "python data_pipeline/features.py"),
]

def log(msg: str):
    """Timestamp + optional file logging."""
    ts = datetime.datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
    line = f"{ts} {msg}"
    print(line)
    if USE_LOG_FILE:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")

def run_step(label: str, cmd: str) -> bool:
    log(f"\n🚀 {label}")
    log(f"→ Running: {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        log(f"❌ Failed: {label} (exit code {result.returncode})")
        return False
    log(f"✅ Success: {label}")
    return True

def main():
    # fresh log
    if USE_LOG_FILE:
        os.makedirs(LOG_DIR, exist_ok=True)
        open(LOG_PATH, "w", encoding="utf-8").close()

    log("🔧 Starting ReviewRadar CI/CD Pipeline\n")

    for label, cmd in STEPS:
        if not run_step(label, cmd):
            log("🛑 Pipeline stopped due to error.\n")
            sys.exit(1)

    log("\n🎉 Pipeline completed successfully!")

if __name__ == "__main__":
    main()
 