from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

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

ID2LABEL = {
    0: "기쁨",
    1: "당황",
    2: "분노",
    3: "불안",
    4: "상처",
    5: "슬픔"
}

KOREAN_TO_ENGLISH = {
    "기쁨": "joy",
    "당황": "surprise",
    "분노": "anger",
    "불안": "fear",
    "상처": "hurt",  
    "슬픔": "sadness"
}

app = FastAPI()

@app.get("/")
async def health_check():
    return {"status": "awake"}

class TextRequest(BaseModel):
    text: str

LINKS = {
    "joy": "https://gift.kakao.com/product/10618518",
    "surprise": "https://gift.kakao.com/product/11561204",
    "anger": "https://gift.kakao.com/product/9314157",
    "fear": "https://gift.kakao.com/product/4764917",
    "sadness": "https://gift.kakao.com/product/11914005",
    "hurt": "https://gift.kakao.com/product/11914005" 
}

def predict_emotion_probabilities(sentence):
    if model is None: 
        return "joy", {"joy": 100.0, "sadness": 0.0} 

    inputs = tokenizer(sentence, return_tensors="pt", truncation=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = F.softmax(logits, dim=1)[0]


    prob_dict = {}
    for i, prob in enumerate(probs):
        kor_label = ID2LABEL[i]
        eng_label = KOREAN_TO_ENGLISH.get(kor_label)
        if eng_label:
            prob_dict[eng_label] = round(prob.item() * 100, 2)
            

    sorted_probs = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
    top_emotion = sorted_probs[0][0]

    return top_emotion, prob_dict


@app.post("/analyze")
async def analyze(request: TextRequest):
    input_text = request.text.strip()
    
    top_emotion, all_probs = predict_emotion_probabilities(input_text)
    
    link = LINKS.get(top_emotion, LINKS["sadness"])

    return {
        "text": input_text,
        "emotion": top_emotion, 
        "probabilities": all_probs, 
        "link": link
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)