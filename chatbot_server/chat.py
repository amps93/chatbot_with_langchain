from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 로컬환경에서 만들어놓은 EEVE 모델 사용
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
        # 최근 사용자와의 대화기록을 기억하기 위한 명령어
    ]
)

# LangChain 표현식
chain = prompt | llm_eeve | StrOutputParser()
# prompt : LLM 설정
# llm_eeve : LLM 종류
# StrOutputParser() : 채팅 메시지를 문자열로 변환하는 간단한 출력 구문 분석기
