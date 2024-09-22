from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import gradio as gr

load_dotenv()


def respond(message, chat_history):
    response = ChatOpenAI(
        model='gpt-3.5-turbo',
        max_tokens=256,
        temperature=0
    )

    print(response)

    bot_message = response.invoke(message).content
    chat_history.append((message, bot_message))

    return "", chat_history


with gr.Blocks() as demo:
    chatbot = gr.Chatbot(label='채팅창')
    msg = gr.Textbox(label='입력')
    clear = gr.Button('초기화')

    msg.submit(respond, [msg, chatbot], [msg, chatbot])  # 텍스트박스에 메시지를 입력하고 제출하면 respond 함수가 호출
    clear.click(lambda: None, None, chatbot, queue=False)  # '초기화' 버튼 클릭 시 채팅 기록 초기화

demo.launch(debug=True)
