from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv()


def respond(message, specialist, chat_history):
    response = ChatOpenAI(
        model='gpt-3.5-turbo',
        max_tokens=256,
        temperature=0
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", "이 시스템은 {specialist} 질문에 답변할 수 있습니다."),
        ("user", "{user_input}"),
    ])

    chain = chat_prompt | response | StrOutputParser()

    if specialist == '없음':
        bot_message = '분야를 선택해주세요'
    else:
        bot_message = chain.invoke({'user_input': message, 'specialist': specialist})

    chat_history.append((message, bot_message))

    return "", chat_history


with gr.Blocks() as demo:
    specialist = gr.Dropdown(label='질문 분야 선택', value='없음', choices=['없음', '천문학', '여행', '음악'])
    chatbot = gr.Chatbot(label='채팅창')
    msg = gr.Textbox(label='입력')
    clear = gr.Button('초기화')

    msg.submit(respond, [msg, specialist, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출
    clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼 클릭 시 채팅 기록 초기화

demo.launch(debug=True)
