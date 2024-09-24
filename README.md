# Chatbot with Langchain

2024.09.16~

## 1. chatbot_gradio

아래 포스트의 도움을 받았습니다.

> [랭체인 러닝데이 | 4. 랭체인 LLM와 그라디오 챗봇 연동하기](https://aifactory.space/task/2446/discussion/400)
> 
> [랭체인 러닝데이 | 5. 랭체인 챗모델(ChatGPT)과 그라디오 챗봇 연동하기](https://aifactory.space/task/2446/discussion/401)
> 
> [랭체인 러닝데이 | 6. PDF 파일기반 질의응답 챗봇(랭체인, 그라디오, ChatGPT)](https://aifactory.space/task/2446/discussion/402)

### 1. 일반 챗봇 - default_chat.py

gpt-3.5-turbo 모델을 활용한 특별한 기능 없는 챗봇

langchain과 gradio 연동 연습에 활용

### 2. 전문가 챗봇 - chatbot_with_prompt.py

prompt를 지정해 특정 분야의 전문가 처럼 답변하는 챗봇

프롬프트를 지정하는 것이 실제 답변의 전문성에 영향이 있는지는 확인해봐야함

### 3. PDF 파일 기반 챗봇 - chatbot_pdf.py

사용자가 PDF 파일을 업로드 하면 PDF 기반으로 답변하는 챗봇

PyMuPDFLoader, RecursiveCharacterTextSplitter, OpenAIEmbeddings, FAISS, retriever 등 RAG를 적극 활용

기존 코드에서 개선점

1. 초기화 버튼 클릭 시 업로드한 PDF 정보도 초기화

2. faiss를 활용해 vectorstore 형태로 저장하여 답변하여 답변 마다 PDF를 로딩하지 않음. 이로 인해 답변 시간 단축. 새로운 PDF를 업로드 할 시에만 다시 로딩. 

3. 



## 2. chatbot_server - using Ollama

llama의 한국어 튜닝 모델인 EEVE 모델 사용하여 챗봇 구축

아래 포스트의 도움을 받았습니다

> [오픈소스 EEVE 모델 로컬환경에서 실행하기 2탄 (LangChain / LangServe / ngrok)](https://marcus-story.tistory.com/30)

### 1. Langchain/LangServe 설치

```python
pipenv install 'langserve[all]'
pipenv install langchain-cli
```

### 2. 프로젝트 생성

```python
langchain app new [파일명]
```

### 3. 코드 작성

#### 1. app.py

```python
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

llm_eeve = Ollama(model='eeve')

# Prompt 설정
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI Assistant Your name is 'Marcus bot'. "
            "You are a smart and humorous and intellectual. You always answer in Korean.",
        ),

        MessagesPlaceholder(variable_name="messages"),
    ]
)

# 체인 생성
chain = prompt | llm_eeve | StrOutputParser()
```

#### 2. server.py

```python
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Union
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from chat import chain


app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 모든 출처의 요청 허용
    allow_credentials=True,  # 교차 출처 요청에 대해 쿠키가 지원되어야 함
    allow_methods=["*"],  # 교차 출처 요청에 허용되어야 하는 HTTP 메소드 목록
    allow_headers=["*"],  # 교차 출처 요청에 대해 지원되어야 하는 HTTP 요청 헤더 목록
    expose_headers=["*"],  # 브라우저에서 액세스할 수 있어야 하는 응답 헤더
)


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/chat/playground")


class Input(BaseModel):
    """Input for the chat endpoint.""" 
    messages: List[Union[HumanMessage, AIMessage, SystemMessage]] = Field(
        ..., description="the messages representing the current converation.", 
    )


# 수정된 코드
add_routes(app,
           chain.with_types(input_type=Input),
           path="/chat",
           enable_feedback_endpoint=True,
           enable_public_trace_link_endpoint=True,
           playground_type="chat",
)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

```

#### 3. 코드 실행

```
python server.py
```

### 4. ngrok 사용해 로컬 서버 포워딩

1. ngrok 다운로드 - [Download](https://ngrok.com/download)

2. 환경 변수 설정 - ngrok.exe 파일 있는 경로 추가
   
   1. 환경 설정 > 편집 > Path > 추가 > C:\경로\ngrok

3. ngrok_url 확인 - 아래 주소에서 Domains에 들어가면 확인 가능
   
   1. https://dashboard.ngrok.com/
