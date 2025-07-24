# scripts/run_schema.py
import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

engine = create_engine(
    f"postgresql://{os.getenv('DB_USER')}:{os.getenv('DB_PASS')}@"
    f"{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/"
    f"{os.getenv('DB_NAME')}"
)

with engine.begin() as conn:
    ddl = open("database/schema.sql", "r", encoding="utf-8").read()
    conn.execute(text(ddl))

print("✅ Schema applied!")
