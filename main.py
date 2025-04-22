from fastapi import FastAPI
from pydantic import BaseModel
import os
import google.generativeai as genai
from dotenv import load_dotenv
import logging

load_dotenv()

app = FastAPI()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

genai.configure(api_key=os.getenv("GEMINI_API"))
model = genai.GenerativeModel('gemini-2.0-flash')  # 또는 gemini-2.0-pro, gemini-1.5-pro 원하는 거 선택

class DiaryRequest(BaseModel):
    content: str

class EmotionResponse(BaseModel):
    emotion: str

def generate_with_google(prompt: str) -> str:
    try:
        response = model.generate_content(prompt)
        return response.text if response else "GEMINI API error"
    except Exception as e:
        logger.error(f"Google API 호출 중 오류 발생: {e}")
        return "API 호출 중 오류 발생"

@app.post("/analyze_emotion", response_model=EmotionResponse)
async def analyze_emotion(request: DiaryRequest):
    prompt = f"""
너는 감정 분석가야. 아래의 일기 내용을 읽고, 감정을 하나의 단어로 분석해줘.
(예: 행복, 슬픔, 분노, 불안 등)

일기 내용:
\"\"\"{request.content}\"\"\"

답변 형식:
감정: [감정단어]
번역: [영어 번역 결과]
    """

    raw_text = generate_with_google(prompt)

    if "감정:" in raw_text:
        emotion_line = raw_text.split("감정:")[-1].split("\n")[0].strip()
        emotion = emotion_line
    else:
        emotion = raw_text.strip()

    return EmotionResponse(emotion=emotion)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
