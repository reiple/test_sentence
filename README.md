# 실행을 위한 준비
  * Ollama 설치
  * ollama pull nomic-embed-text

# 실행 방법
  * pip install -r requirements.txt
  * uvicorn main:app --reload
  * http://localhost:8000/docs 접속

# 샘플 데이터
```
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
```
# API
## /find_similar
  * query: 코딩
  * top_k: 3
  * 결과
```json
{
  "results": [
    {
      "sentence": "코딩 공부가 재미있네요",
      "similarity": 0.797939699170711
    },
    {
      "sentence": "커피 한잔 하실래요?",
      "similarity": 0.697889152638606
    },
    {
      "sentence": "점심 메뉴 추천해주세요",
      "similarity": 0.5765961329603881
    }
  ]
}
```

## /word_cloud
![image](https://github.com/user-attachments/assets/a4f6edeb-bea1-48ab-a59e-a8fc6b4ce63d)
