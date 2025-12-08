import os
from fastapi import FastAPI, HTTPException, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime
import random
import traceback # 에러 로그용

# DB 주소 설정
DATABASE_URL = "postgresql://lbeul372:mhi6qvmdTSSp2rGpAYX8dA33IMnFwGqm@dpg-d4pqolm3jp1c7395lr6g-a/emo_db"

if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+pg8000://", 1)
elif DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+pg8000://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# === 테이블 정의 ===
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)      
    text = Column(Text)                       
    ai_emotion = Column(String)               
    ai_link = Column(String)
    probabilities = Column(JSON)              
    user_emotion = Column(String, nullable=True) 
    user_link = Column(String, nullable=True)
    is_corrected = Column(Boolean, default=False) 
    created_at = Column(DateTime, default=datetime.now)

class TrainingData(Base):
    __tablename__ = "training_data"
    id = Column(Integer, primary_key=True, index=True)
    text = Column(Text)        
    label = Column(String)       
    created_at = Column(DateTime, default=datetime.now)

# 테이블 생성 (주의: 이미 테이블이 있으면 무시함 -> 그래서 DB 초기화가 필요함)
Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# === 모델 로딩 ===
print("⏳ AI 모델을 로딩 중입니다...")
model_path = "./emotion_model" 
try:
    if os.path.exists(model_path):
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.eval()
        print("✅ 모델 로딩 완료!")
    else:
        print("⚠️ 모델 폴더 없음: 기본 로직으로 동작합니다.")
        model = None
except Exception as e:
    print(f"❌ 모델 로딩 실패! {e}")
    model = None

# === 상수 데이터 ===
ID2LABEL = {0: "기쁨", 1: "당황", 2: "분노", 3: "불안", 4: "상처", 5: "슬픔"}
KOREAN_TO_ENGLISH = {"기쁨": "joy", "당황": "surprise", "분노": "anger", "불안": "fear", "상처": "hurt", "슬픔": "sadness"}

LINKS = {
    "joy": ["https://gift.kakao.com/product/11555336", "https://gift.kakao.com/product/11580485?banner_id=1275&campaign_code=null", "https://gift.kakao.com/product/11780438?banner_id=1275&campaign_code=null"],
    "anger": ["https://gift.kakao.com/product/4764917?banner_id=1275&campaign_code=null", "https://gift.kakao.com/product/12033009?banner_id=1275&campaign_code=null", "https://gift.kakao.com/product/10846674?banner_id=1275&campaign_code=null"],
    "fear": ["https://gift.kakao.com/product/12456517?banner_id=1275&campaign_code=null", "https://gift.kakao.com/product/4602190?banner_id=1275&campaign_code=null", "https://gift.kakao.com/product/8104244?banner_id=1275&campaign_code=null"],
    "sadness": ["https://gift.kakao.com/product/11984533?banner_id=1275&campaign_code=null", "https://gift.kakao.com/product/11190291?banner_id=1275&campaign_code=null", "https://gift.kakao.com/product/8664156?banner_id=1275&campaign_code=null"],
    "surprise": ["https://gift.kakao.com/product/7455892?banner_id=1275&campaign_code=null", "https://gift.kakao.com/product/12165302?banner_id=1275&campaign_code=null", "https://gift.kakao.com/product/11030220?banner_id=1275&campaign_code=null"],
    "hurt": ["https://gift.kakao.com/product/11662431?banner_id=1275&campaign_code=null", "https://gift.kakao.com/product/7264823?banner_id=1275&campaign_code=null", "https://gift.kakao.com/product/7320451?banner_id=1275&campaign_code=null"]
}

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

class AnalyzeRequest(BaseModel):
    text: str
    user_id: str

class ReportRequest(BaseModel):
    log_id: int
    user_label: str

@app.get("/")
async def read_root():
    # 현재 파일 위치 기준으로 index.html 찾기
    file_path = os.path.join(os.path.dirname(__file__), "index.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    return {"error": "index.html file not found inside server"}

@app.post("/analyze")
async def analyze(request: AnalyzeRequest, db: Session = Depends(get_db)):
    try:
        input_text = request.text.strip()
        
        if model:
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
            with torch.no_grad():
                outputs = model(**inputs)
                probs = F.softmax(outputs.logits, dim=1)[0]
            
            prob_dict = {}
            for i, prob in enumerate(probs):
                eng_label = KOREAN_TO_ENGLISH.get(ID2LABEL[i])
                if eng_label:
                    prob_dict[eng_label] = round(prob.item() * 100, 2)
            
            sorted_probs = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
            top_emotion = sorted_probs[0][0]
        else:
            top_emotion = "joy"
            prob_dict = {"joy": 100.0}

        link_list = LINKS.get(top_emotion, LINKS["sadness"])
        link = random.choice(link_list)

        new_log = PredictionLog(
            user_id=request.user_id,
            text=input_text,
            ai_emotion=top_emotion,
            ai_link=link,
            probabilities=prob_dict,
            is_corrected=False
        )
        db.add(new_log)
        db.commit()
        db.refresh(new_log)

        return {
            "log_id": new_log.id,
            "text": input_text,
            "emotion": top_emotion, 
            "probabilities": prob_dict, 
            "link": link,
            "is_corrected": False
        }
    except Exception as e:
        # 에러 발생 시 로그 출력
        print(f"❌ 분석 중 에러 발생: {e}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/report")
async def report_bug(request: ReportRequest, db: Session = Depends(get_db)):
    try:
        log = db.query(PredictionLog).filter(PredictionLog.id == request.log_id).first()
        if not log:
            raise HTTPException(status_code=404, detail="Log not found")
        
        if log.is_corrected:
             return {"message": "Already reported", "emotion": log.user_emotion}

        link_list = LINKS.get(request.user_label, LINKS["sadness"])
        new_link = random.choice(link_list)
        
        log.user_emotion = request.user_label
        log.user_link = new_link
        log.is_corrected = True

        train_data = TrainingData(
            text=log.text,
            label=request.user_label
        )
        db.add(train_data)
        db.commit()
        
        return {
            "message": "Reported",
            "new_emotion": request.user_label,
            "new_link": new_link
        }
    except Exception as e:
        print(f"❌ 리포트 중 에러 발생: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history/{user_id}")
async def get_history(user_id: str, db: Session = Depends(get_db)):
    logs = db.query(PredictionLog)\
             .filter(PredictionLog.user_id == user_id)\
             .order_by(PredictionLog.created_at.desc())\
             .all()
    
    result = []
    for log in logs:
        if log.is_corrected:
            display_emotion = log.user_emotion
            display_link = log.user_link
        else:
            display_emotion = log.ai_emotion
            display_link = log.ai_link
            
        result.append({
            "log_id": log.id,
            "text": log.text,
            "emotion": display_emotion, 
            "link": display_link,       
            "probabilities": log.probabilities,
            "is_corrected": log.is_corrected,
            "date": log.created_at.strftime("%Y-%m-%d %H:%M")
        })
    return result

@app.delete("/history/{log_id}")
async def delete_history(log_id: int, db: Session = Depends(get_db)):
    log = db.query(PredictionLog).filter(PredictionLog.id == log_id).first()
    if log:
        db.delete(log)
        db.commit()
        return {"message": "Deleted"}
    raise HTTPException(status_code=404, detail="Log not found")

@app.get("/admin/reports")
async def get_training_data(db: Session = Depends(get_db)):
    reports = db.query(TrainingData).order_by(TrainingData.created_at.desc()).all()
    return reports

@app.get("/admin/all_logs")
async def get_all_logs(db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).order_by(PredictionLog.created_at.desc()).all()
    
    result = []
    for log in logs:
        if log.is_corrected:
            display_emotion = log.user_emotion
            display_link = log.user_link
        else:
            display_emotion = log.ai_emotion
            display_link = log.ai_link
            
        result.append({
            "log_id": log.id,
            "user_id": log.user_id, 
            "text": log.text,
            "emotion": display_emotion, 
            "link": display_link,       
            "probabilities": log.probabilities,
            "is_corrected": log.is_corrected,
            "date": log.created_at.strftime("%Y-%m-%d %H:%M:%S")
        })
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)