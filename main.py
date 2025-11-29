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
    4: "상처",  # <-- 앱에는 없는 감정
    5: "슬픔"
}

# 앱(Flutter)에서 사용하는 영어 키 매핑 (상처는 제외됨)
KOREAN_TO_ENGLISH = {
    "기쁨": "joy",
    "당황": "surprise",
    "분노": "anger",
    "불안": "fear",
    "슬픔": "sadness"
    # "상처"는 여기에 없음 -> 로직에서 걸러냄
}

# ==========================================
# 2. FastAPI 설정
# ==========================================
app = FastAPI()

class TextRequest(BaseModel):
    text: str

# 선물 링크 (기존 유지)
LINKS = {
    "joy": "https://gift.kakao.com/product/10618518",
    "surprise": "https://gift.kakao.com/product/11561204",
    "anger": "https://gift.kakao.com/product/9314157",
    "fear": "https://gift.kakao.com/product/4764917",
    "sadness": "https://gift.kakao.com/product/11914005"
}

# ==========================================
# 3. 핵심 로직: 2순위 감정 찾기
# ==========================================
def predict_emotion(sentence):
    if model is None: return "기쁨" # 모델 없으면 그냥 기쁨 리턴

    # 1. 추론 (Inference)
    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        # Softmax로 확률(%) 계산
        probs = F.softmax(logits, dim=1)[0]

    # 2. 확률 높은 순서대로 정렬 [(확률, 라벨인덱스), ...]
    # 예: [(0.8, 4='상처'), (0.15, 5='슬픔'), ...]
    probs_list = []
    for i, prob in enumerate(probs):
        probs_list.append((prob.item(), i))
    
    # 확률 높은 순으로 정렬 (내림차순)
    probs_list.sort(key=lambda x: x[0], reverse=True)

    # 3. 앱에서 쓸 수 있는 감정인지 확인 (순서대로 체크)
    final_emotion_kor = "슬픔" # 기본값
    
    for prob, idx in probs_list:
        korean_label = ID2LABEL[idx]
        
        # 만약 이 라벨이 내 앱(영어키 매핑)에 있다면? -> 채택!
        if korean_label in KOREAN_TO_ENGLISH:
            final_emotion_kor = korean_label
            print(f"👉 선택된 감정: {korean_label} (확률: {prob*100:.1f}%)")
            break
        else:
            # 상처 처럼 앱에 없는 라벨이면? -> 패스하고 다음으로 높은 거 봄
            print(f"🚫 스킵된 감정: {korean_label} (앱 미지원)")

    return final_emotion_kor

# ==========================================
# 4. API 엔드포인트
# ==========================================
@app.post("/analyze")
async def analyze(request: TextRequest):
    input_text = request.text.strip()
    
    # AI 예측 수행
    korean_emotion = predict_emotion(input_text)
    
    # 영어로 변환 (위에서 필터링했으므로 무조건 있음)
    english_emotion = KOREAN_TO_ENGLISH[korean_emotion]
    link = LINKS[english_emotion]

    return {
        "text": input_text,
        "emotion": english_emotion,
        "original_emotion": korean_emotion,
        "link": link
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)