# 실행을 위한 준비
  * Ollama 설치
  * ollama pull nomic-embed-text

# 실행 방법
  * pip install -r requirements.txt
  * uvicorn main:app --reload
  * http://localhost:8000/docs 접속

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
