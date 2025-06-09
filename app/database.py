from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os

from app.config import settings

# 환경별 DB URL 설정
DATABASE_URL = f"mysql+pymysql://{settings.DB_USER}:{settings.DB_PASSWORD}@{settings.DB_HOST}:{settings.DB_PORT}/{settings.DB_NAME}"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def export_tables_to_csv():
    """DB 테이블들을 CSV로 export"""
    tables = ["plans", "services", "coupons"]
    os.makedirs("./app/data/pricing", exist_ok=True)

    for table in tables:
        df = pd.read_sql(f"SELECT * FROM {table}", engine)
        csv_path = f"./app/data/pricing/{table}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"{table} -> {csv_path}")