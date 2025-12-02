from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ==========================================
# 1. AI 모델 로드
# ==========================================
print("⏳ AI 모델을 로딩 중입니다...")
model_path = "./emotion_model"  # 폴더 이름 꼭 확인하세요!

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.eval()
    print("✅ 모델 로딩 완료!")
except Exception as e:
    print(f"❌ 모델 로딩 실패! {e}")
    # 실패하면 더미 모델로라도 작동하게 예외처리 (급하니까)
    model = None

# 팀원 모델의 6개 라벨 (순서 중요! 팀원 코드 기준 0~5)
ID2LABEL = {
    0: "기쁨",
    1: "당황",
    2: "분노",
    3: "불안",
    4: "상처",
    5: "슬픔"
}

# 앱(Flutter)에서 사용하는 영어 키 매핑 (상처 추가됨)
KOREAN_TO_ENGLISH = {
    "기쁨": "joy",
    "당황": "surprise",
    "분노": "anger",
    "불안": "fear",
    "상처": "hurt",  # 새로 추가됨
    "슬픔": "sadness"
}

# ==========================================
# 2. FastAPI 설정
# ==========================================
app = FastAPI()

class TextRequest(BaseModel):
    text: str

# 선물 링크 (기존 유지, hurt는 sadness 링크 사용)
LINKS = {
    "joy": "https://gift.kakao.com/product/10618518",
    "surprise": "https://gift.kakao.com/product/11561204",
    "anger": "https://gift.kakao.com/product/9314157",
    "fear": "https://gift.kakao.com/product/4764917",
    "sadness": "https://gift.kakao.com/product/11914005",
    "hurt": "https://gift.kakao.com/product/11914005" # 임시로 슬픔 링크 사용
}

# ==========================================
# 3. 핵심 로직: 감정 확률 계산 및 상위 감정 선택
# ==========================================
def predict_emotion_probabilities(sentence):
    if model is None: 
        # 모델 로딩 실패 시 기본값 반환
        return "joy", {"joy": 100.0, "sadness": 0.0} 

    # 1. 추론 (Inference)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Softmax로 확률(%) 계산
        probs = F.softmax(logits, dim=1)[0]

    # 2. {영어 감정: 확률%} 딕셔너리 생성
    prob_dict = {}
    for i, prob in enumerate(probs):
        kor_label = ID2LABEL[i]
        eng_label = KOREAN_TO_ENGLISH.get(kor_label)
        if eng_label:
            prob_dict[eng_label] = round(prob.item() * 100, 2)
            
    # 3. 확률이 가장 높은 감정 찾기
    # 확률 내림차순 정렬
    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    top_emotion = sorted_probs[0][0]

    return top_emotion, prob_dict

# ==========================================
# 4. API 엔드포인트
# ==========================================
@app.post("/analyze")
async def analyze(request: TextRequest):
    input_text = request.text.strip()
    
    # AI 예측 수행 (최상위 감정, 전체 확률 정보)
    top_emotion, all_probs = predict_emotion_probabilities(input_text)
    
    # 선물 링크 가져오기
    link = LINKS.get(top_emotion, LINKS["sadness"])

    return {
        "text": input_text,
        "emotion": top_emotion,     # UI 표시용 메인 감정
        "probabilities": all_probs, # '더 보기'용 전체 확률 정보
        "link": link
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)