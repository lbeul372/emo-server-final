from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
import uvicorn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from datetime import datetime

# ==========================================
# 1. DB 설정 (여기에 Render DB 주소 입력!)
# ==========================================
# 예: "postgres://emo_user:password@.../emo_db"
DATABASE_URL = "postgresql://lbeul372:mhi6qvmdTSSp2rGpAYX8dA33IMnFwGqm@dpg-d4pqolm3jp1c7395lr6g-a/emo_db"

# Render의 주소(postgres://)를 SQLAlchemy용(postgresql://)으로 자동 변환
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql+pg8000://", 1)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# ==========================================
# 2. DB 모델 정의 (최종 확정된 설계)
# ==========================================
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)      # 사용자 UUID
    text = Column(Text)                       # 입력 텍스트
    
    # [AI 원본 데이터 - 보존됨]
    ai_emotion = Column(String)               
    ai_link = Column(String)
    probabilities = Column(JSON)              # 6가지 확률 (JSON)

    # [사용자 수정 데이터 - 리포트 시 채워짐]
    user_emotion = Column(String, nullable=True) 
    user_link = Column(String, nullable=True)
    
    # [상태 값]
    is_corrected = Column(Boolean, default=False) # 리포트 여부
    created_at = Column(DateTime, default=datetime.now)

# DB 테이블 자동 생성 (서버 시작 시)
Base.metadata.create_all(bind=engine)

# DB 세션 의존성 함수
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 3. AI 모델 및 데이터 준비
# ==========================================
print("⏳ AI 모델을 로딩 중입니다...")
model_path = "./emotion_model" 

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print("✅ 모델 로딩 완료!")
except Exception as e:
    print(f"❌ 모델 로딩 실패! {e}")
    model = None

ID2LABEL = {0: "기쁨", 1: "당황", 2: "분노", 3: "불안", 4: "상처", 5: "슬픔"}
KOREAN_TO_ENGLISH = {"기쁨": "joy", "당황": "surprise", "분노": "anger", "불안": "fear", "상처": "hurt", "슬픔": "sadness"}
LINKS = {
    "joy": "https://gift.kakao.com/product/10618518",
    "surprise": "https://gift.kakao.com/product/11561204",
    "anger": "https://gift.kakao.com/product/9314157",
    "fear": "https://gift.kakao.com/product/4764917",
    "sadness": "https://gift.kakao.com/product/11914005",
    "hurt": "https://gift.kakao.com/product/11914005" 
}

app = FastAPI()

# ==========================================
# 4. 요청/응답 스키마 (Pydantic)
# ==========================================
class AnalyzeRequest(BaseModel):
    text: str
    user_id: str  # Flutter에서 UUID 받음

class ReportRequest(BaseModel):
    log_id: int   # 수정할 기록의 ID
    user_label: str # 사용자가 선택한 정답 감정

# ==========================================
# 5. API 엔드포인트
# ==========================================

# [기능 0] 서버 깨우기 (Cron-job용)
@app.get("/")
async def health_check():
    return {"status": "awake"}

# [기능 1] 감정 분석 요청 (AI -> DB 저장)
@app.post("/analyze")
async def analyze(request: AnalyzeRequest, db: Session = Depends(get_db)):
    input_text = request.text.strip()
    
    # 1. AI 예측 수행
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
        # 모델 로드 실패 시 기본값 (테스트용)
        top_emotion = "joy"
        prob_dict = {"joy": 100.0}

    link = LINKS.get(top_emotion, LINKS["sadness"])

    # 2. DB에 'AI 원본 데이터'로 저장
    new_log = PredictionLog(
        user_id=request.user_id,
        text=input_text,
        ai_emotion=top_emotion,
        ai_link=link,
        probabilities=prob_dict,
        is_corrected=False # 초기엔 수정 안 됨
    )
    db.add(new_log)
    db.commit()
    db.refresh(new_log)

    # 3. 결과 반환 (Flutter가 화면 그리기용)
    return {
        "log_id": new_log.id, # 중요: 나중에 리포트할 때 이 번호 씀
        "text": input_text,
        "emotion": top_emotion, 
        "probabilities": prob_dict, 
        "link": link,
        "is_corrected": False
    }

# [기능 2] 버그 리포트 (사용자 수정 -> DB 업데이트)
@app.post("/report")
async def report_bug(request: ReportRequest, db: Session = Depends(get_db)):
    # 1. 기록 찾기
    log = db.query(PredictionLog).filter(PredictionLog.id == request.log_id).first()
    if not log:
        raise HTTPException(status_code=404, detail="Log not found")
    
    # 2. 이미 리포트된 건지 확인 (중복 방지)
    if log.is_corrected:
         return {"message": "Already reported", "emotion": log.user_emotion}

    # 3. 사용자 의견 반영 (AI 원본은 건드리지 않음!)
    new_link = LINKS.get(request.user_label, LINKS["sadness"])
    
    log.user_emotion = request.user_label
    log.user_link = new_link
    log.is_corrected = True # 수정됨 표시
    
    db.commit()
    
    return {
        "message": "Reported",
        "new_emotion": request.user_label,
        "new_link": new_link
    }

# [기능 3] 히스토리 불러오기 (사용자별)
@app.get("/history/{user_id}")
async def get_history(user_id: str, db: Session = Depends(get_db)):
    # 최신순으로 정렬해서 가져오기
    logs = db.query(PredictionLog)\
             .filter(PredictionLog.user_id == user_id)\
             .order_by(PredictionLog.created_at.desc())\
             .all()
    
    # Flutter가 보기 편하게 데이터 가공
    result = []
    for log in logs:
        # 수정된 기록이면 -> user 데이터 사용
        if log.is_corrected:
            display_emotion = log.user_emotion
            display_link = log.user_link
        # 원본 기록이면 -> ai 데이터 사용
        else:
            display_emotion = log.ai_emotion
            display_link = log.ai_link
            
        result.append({
            "log_id": log.id,
            "text": log.text,
            "emotion": display_emotion, # 화면에 보여줄 최종 감정
            "link": display_link,       # 화면에 보여줄 최종 링크
            "probabilities": log.probabilities,
            "is_corrected": log.is_corrected,
            "date": log.created_at.strftime("%Y-%m-%d %H:%M") # 날짜 예쁘게
        })
        
    return result

# [기능 4] 히스토리 삭제
@app.delete("/history/{log_id}")
async def delete_history(log_id: int, db: Session = Depends(get_db)):
    log = db.query(PredictionLog).filter(PredictionLog.id == log_id).first()
    if log:
        db.delete(log)
        db.commit()
        return {"message": "Deleted"}
    raise HTTPException(status_code=404, detail="Log not found")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)