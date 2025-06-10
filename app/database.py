from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import sessionmaker
import pandas as pd
import os

from app.config import settings

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

def main():
    """DB 초기화 및 CSV export"""
    Base.metadata.create_all(bind=engine)
    print("DB 초기화 완료")
    
    export_tables_to_csv()
    print("CSV export 완료")
    
    print("모든 작업이 완료되었습니다.")
if __name__ == "__main__":
    main()