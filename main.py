from fastapi import FastAPI, Form
import numpy as np
from typing import List, Dict
import uvicorn
import requests
import json
from collections import Counter
import re
from pydantic import BaseModel
from typing import List, Dict
from wordcloud import WordCloud
import io
from fastapi.responses import StreamingResponse
import matplotlib.pyplot as plt
from PIL import Image

app = FastAPI(
    title="문장 임베딩 및 단어 분석 API",
    description="문장 임베딩을 이용한 유사도 검색과 단어 빈도 분석을 제공하는 API",
    version="1.0.0"
)

# Ollama API endpoint
OLLAMA_API = "http://localhost:11434"

# 예시 문장들 (실제 사용시에는 데이터베이스나 파일에서 로드하면 됩니다)
sample_sentences = [
    "오늘 날씨가 정말 좋네요",
    "내일은 비가 올 것 같아요",
    "주말에 영화 보러 갈까요?",
    "이 책은 정말 재미있어요",
    "점심 메뉴 추천해주세요",
    "운동하러 가야겠어요",
    "새로운 프로젝트를 시작했어요",
    "커피 한잔 하실래요?",
    "여행 계획을 세우고 있어요",
    "코딩 공부가 재미있네요"
]

# 응답 모델 정의
class WordFrequency(BaseModel):
    word: str
    frequency: int

class SentenceFrequency(BaseModel):
    sentence: str
    word_frequency: Dict[str, int]

class WordStatisticsResponse(BaseModel):
    total_words: int
    word_list: List[Dict[str, int]]
    total_frequency: Dict[str, int]
    sentence_frequencies: List[Dict[str, Dict[str, int]]]

class SimilarSentence(BaseModel):
    sentence: str
    similarity: float
    word_frequency: Dict[str, int]

class FindSimilarResponse(BaseModel):
    query_word_frequency: Dict[str, int]
    results: List[Dict[str, object]]

def get_embedding(text: str) -> List[float]:
    """Ollama를 사용하여 텍스트의 임베딩을 얻습니다."""
    response = requests.post(
        f"{OLLAMA_API}/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )
    response_json = response.json()
    print("API Response:", response_json)  # 응답 구조 확인을 위한 출력
    
    # Ollama API는 'embedding' 키를 사용합니다
    return response_json["embedding"]

def get_word_frequency(text: str) -> Dict[str, int]:
    """텍스트에서 단어 빈도수를 계산합니다."""
    # 한글, 영문, 숫자만 남기고 나머지는 공백으로 변경
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    # 공백을 기준으로 단어 분리
    words = text.split()
    # 단어 빈도수 계산
    return dict(Counter(words))

# 샘플 문장들의 임베딩을 미리 계산
sample_embeddings = [get_embedding(sentence) for sentence in sample_sentences]

@app.post("/find_similar")
async def find_similar(query: str = Form(...), top_k: int = Form(3)):
    # 입력된 쿼리의 임베딩 계산
    query_embedding = get_embedding(query)
    
    # 코사인 유사도 계산
    query_embedding = np.array(query_embedding)
    sample_embeddings_array = np.array(sample_embeddings)
    
    similarities = np.dot(sample_embeddings_array, query_embedding) / (
        np.linalg.norm(sample_embeddings_array, axis=1) * np.linalg.norm(query_embedding)
    )
    
    # 가장 유사한 top_k개의 문장 찾기
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    results = []
    for idx in top_indices:
        results.append({
            "sentence": sample_sentences[idx],
            "similarity": float(similarities[idx])
        })
    
    return {"results": results}

@app.get("/word_statistics", response_model=WordStatisticsResponse, tags=["통계"])
async def get_word_statistics():
    """
    전체 샘플 문장들의 단어 통계를 반환합니다.
    
    Returns:
        - total_words: 전체 고유 단어 수
        - word_list: 전체 단어 목록과 각 단어의 빈도수 (빈도수 순으로 정렬)
        - total_frequency: 전체 단어 빈도수 맵
        - sentence_frequencies: 각 문장별 단어 빈도수
    """
    # 전체 단어 빈도수
    total_frequency = Counter()
    
    # 각 문장별 단어 빈도수 및 전체 빈도수 계산
    sentence_frequencies = []
    for sentence in sample_sentences:
        word_freq = get_word_frequency(sentence)
        sentence_frequencies.append({
            "sentence": sentence,
            "word_frequency": word_freq
        })
        total_frequency.update(word_freq)
    
    # 전체 단어 목록 (빈도수 순으로 정렬)
    word_list = sorted(
        [{"word": word, "frequency": freq} 
         for word, freq in total_frequency.items()],
        key=lambda x: x["frequency"],
        reverse=True
    )
    
    return {
        "total_words": len(total_frequency),
        "word_list": word_list,
        "total_frequency": dict(total_frequency),
        "sentence_frequencies": sentence_frequencies
    }

@app.get("/word_cloud", tags=["시각화"])
async def generate_word_cloud():
    """
    단어 빈도수를 기반으로 워드 클라우드 이미지를 생성합니다.
    
    Returns:
        PNG 형식의 워드 클라우드 이미지
    """
    # 전체 단어 빈도수 계산
    total_frequency = Counter()
    for sentence in sample_sentences:
        word_freq = get_word_frequency(sentence)
        total_frequency.update(word_freq)
    
    # 워드 클라우드 생성
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white',
        font_path='C:/Windows/Fonts/malgun.ttf',  # 한글 폰트 사용
        min_font_size=10,
        max_font_size=100
    )
    
    # 단어 빈도수 데이터로 워드 클라우드 생성
    wordcloud.generate_from_frequencies(total_frequency)
    
    # 이미지를 바이트로 변환
    img_bytes = io.BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.savefig(img_bytes, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    img_bytes.seek(0)
    
    return StreamingResponse(img_bytes, media_type="image/png")

@app.get("/", tags=["홈"])
async def read_root():
    return {"message": "Ollama를 이용한 문장 유사도 검색 API에 오신 것을 환영합니다!"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
