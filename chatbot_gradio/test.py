from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import gradio as gr
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

load_dotenv()


class PDFChatbot:
    def __init__(self):
        self.vector_store = None
        self.chain = None
        self.db_path = './db/faiss'
        self.previous_filename = None

    def process_pdf(self, file):
        loader = PyMuPDFLoader(file)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=0,
            length_function=len
        )

        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()

        # 벡터 스토어 생성 후 저장
        self.vector_store = FAISS.from_documents(texts, embeddings)
        self.vector_store.save_local(self.db_path)  # FAISS 데이터베이스를 로컬에 저장

        # 검색 도구 선언
        retriever = self.vector_store.as_retriever(search_kwargs={"k": 2})

        # 모델 및 체인 선언
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

        # 프롬프트 템플릿 생성
        system_template = """Use the following pieces of context to answer the users question shortly.
        Given the following summaries of a long document and a question, create a final answer with references ("SOURCES"), 
        use "SOURCES" in capital letters regardless of the number of sources.
        If you don't know the answer, just say that "I don't know", don't try to make up an answer.
        ----------------
        {summaries}

        You MUST answer in Korean and in Markdown format:"""

        messages = [
            SystemMessagePromptTemplate.from_template(system_template),
            HumanMessagePromptTemplate.from_template("{question}")
        ]

        prompt = ChatPromptTemplate.from_messages(messages)
        chain_type_kwargs = {"prompt": prompt}

        '''
        체인 구축
        RetrievalQAWithSourcesChain: 질의에 대해 문서 저장소에서 문서를 검색, 정보 기반 응답 생성하는 체인
        parameter
        * llm: 언어 모델
        * chain_type: 체인의 유형. 디폴트 stuff(검색된 문서를 요약하여 답변 생성). 
        stuff, map_reduce(답변을 생성할 때 문서들을 개별적으로 처리하고 합쳐서 답변 생성), 또는 refine(답변을 반복적으로 개선) 중 하나 선택
        * retriever: 검색 도구
        * return_source_documents: 답변과 함께 검색된 문서 자체를 반환할지 여부를 설정
        '''
        self.chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )

    def respond(self, file, message, chat_history):
        if file is None:
            return "파일이 업로드되지 않았습니다. PDF 파일을 업로드해 주세요.", chat_history

        current_filename = file.name

        # 파일이 새로 업로드 될 때만 pdf 다시 로드
        if self.previous_filename is None:
            self.previous_filename = current_filename
            self.process_pdf(file)
        elif self.previous_filename != current_filename:
            self.previous_filename = current_filename
            self.process_pdf(file)

        # 질문에 대한 답변 생성
        result = self.chain.invoke({'question': message})
        bot_message = result['answer']

        # 출처 정보 추가
        for i, doc in enumerate(result['source_documents']):
            bot_message += f' [{i + 1}] {doc.metadata["source"]} (페이지: {doc.metadata["page"]}) '

        chat_history.append((message, bot_message))

        return "", chat_history


# Gradio 인터페이스 정의
pdf_chatbot = PDFChatbot()


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label="채팅창")
    file = gr.File(label="PDF 파일 업로드")
    msg = gr.Textbox(label="질문 입력")
    clear = gr.Button("초기화")

    msg.submit(pdf_chatbot.respond, [file, msg, chatbot], [msg, chatbot])
    clear.click(lambda: (None, None), None, [chatbot, file], queue=False)

demo.launch(debug=True)
