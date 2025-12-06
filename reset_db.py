# reset_db.py
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base

DATABASE_URL = "postgresql://lbeul372:mhi6qvmdTSSp2rGpAYX8dA33IMnFwGqm@dpg-d4pqolm3jp1c7395lr6g-a/emo_db"

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+pg8000://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+pg8000://", 1)

engine = create_engine(DATABASE_URL)
Base = declarative_base()

print("🔄 데이터베이스 구조를 갱신합니다...")
try:
    Base.metadata.drop_all(bind=engine)
    print("🗑️ 기존 DB 삭제 완료")
    
    print("✅ 초기화 끝! 이제 배포 완료되면 정상 작동합니다.")
except Exception as e:
    print(f"❌ 오류: {e}")