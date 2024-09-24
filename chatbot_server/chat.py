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

