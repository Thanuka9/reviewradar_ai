# ReviewRadar AI – Yelp Dataset

ReviewRadar AI is an end-to-end review intelligence platform. This repo contains the full pipeline to process the Yelp Open Dataset: load → clean → store → analyze → extract features → generate insights.

## 📊 Modules Covered

- `scraper.py` – Load raw Yelp JSON into PostgreSQL
- `map_businesses.py` – Create lookup tables: categories & geo
- `run_schema.py` – Apply schema enhancements to `users` table
- `eda.py` – Exploratory Data Analysis with charts + markdown report
- `features.py` – Generate `review_features`, `user_features`, `business_features`

---

## ⚙️ Setup Instructions

### 1. Clone & Setup Environment

```bash
git clone <repo_url>
cd reviewradar_ai
python -m venv .venv
source .venv/bin/activate  # or .\.venv\Scripts\Activate.ps1 on Windows
pip install -r requirements.txt
2. Create .env
ini
Copy
Edit
DB_USER=review_user
DB_PASS=root
DB_HOST=localhost
DB_PORT=5432
DB_NAME=reviews
🏗️ Database Setup
A. Start PostgreSQL and Create DB
Create a PostgreSQL database named reviews. Add a user review_user with the password root.

B. Apply Schema
bash
Copy
Edit
python run_schema.py
This will add extended columns to the users table (fans, elite, compliments, etc.).

📥 Load Yelp Dataset
Download JSON files from: https://www.yelp.com/dataset

Place them in:

pgsql
Copy
Edit
data/raw/
├── yelp_academic_dataset_business.json
├── yelp_academic_dataset_user.json
└── yelp_academic_dataset_review.json
Then run:

bash
Copy
Edit
python scraper.py
This will populate the businesses, users, and reviews tables.

🗺️ Map Businesses (Categories + Geo)
bash
Copy
Edit
python map_businesses.py
Creates:

business_categories: one row per (business, category)

business_geo: business_id ↔ latitude/longitude

📈 Exploratory Data Analysis
bash
Copy
Edit
python eda.py
Generates:

13 key plots (star ratings, length, trends, sentiment, user retention)

Markdown report: outputs/eda_report.md

PNGs: under outputs/ folder

🧠 Feature Generation
bash
Copy
Edit
python features.py
This builds:

1. review_features
Sentiment (VADER)

TF-IDF terms

Keyword counts

Time features (DOW, hour, holidays)

Geo clusters (KMeans)

Rolling user rating averages (3m, 6m, 12m)

One-hot encoded top 10 categories

2. user_features
Aggregated review behavior (e.g., review counts, average sentiment)

3. business_features
Derived from joins with reviews and categories

It also saves:

outputs/scaler.pkl — StandardScaler for numerical features

outputs/features.log — Execution log

✅ Outputs Summary
Directory: outputs/

*.png – All plots from EDA and feature script

eda_report.md – Full Markdown report

features.log – Log from feature generation

scaler.pkl – Scaler for downstream ML models

🔮 Next Steps
Train sentiment classifier using review_features

Perform topic modeling using BERTopic/LDA

Build a recommender system

Add Streamlit dashboard

Integrate OpenAI for summarization + Q&A

📁 Project Structure
bash
Copy
Edit
reviewradar_ai/
├── data_pipeline/
│   ├── scraper.py
│   ├── map_businesses.py
│   ├── features.py
│   ├── run_schema.py
│
├── llm/
│   ├── summarizer.py         # (planned)
│   ├── rag_qa.py             # (planned)
│
├── dashboard/
│   └── streamlit_app.py      # (planned)
│
├── database/
│   └── schema.sql
│
├── notebooks/
│   └── eda.py
│
├── outputs/
│   ├── *.png, eda_report.md, features.log, scaler.pkl
│
├── requirements.txt
└── .env
Built with 💡 by ReviewRadar AI

vbnet
Copy
Edit

Would you like me to overwrite your original `readme.md` with this updated version?
```
