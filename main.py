import streamlit as st

from langchain.memory import ConversationBufferMemory
from utils import qa_agent


st.title("PDF問答系統")

with st.sidebar:
    openai_api_key = st.text_input("請輸入openai_api_key：", type="password")
    st.markdown("[獲取OpenAI API key](https://platform.openai.com/account/api-keys)")

if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True,
        memory_key="chat_history",
        output_key="answer"
    )

uploaded_file = st.file_uploader("上傳你的PDF文件：", type="pdf")
question = st.text_input("對PDF的内容進行提問", disabled=not uploaded_file)

if uploaded_file and question and not openai_api_key:
    st.info("請輸入你的OpenAI API_kEY")

if uploaded_file and question and openai_api_key:
    with st.spinner("AI正在思考中，請稍等..."):
        response = qa_agent(openai_api_key, st.session_state["memory"],
                            uploaded_file, question)
    st.write("### 答案")
    st.write(response["answer"])
    st.session_state["chat_history"] = response["chat_history"]

if "chat_history" in st.session_state:
    with st.expander("歷史消息"):
        for i in range(0, len(st.session_state["chat_history"]), 2):
            human_message = st.session_state["chat_history"][i]
            ai_message = st.session_state["chat_history"][i+1]
            st.write(human_message.content)
            st.write(ai_message.content)
            print(len(st.session_state["chat_history"]), i)
            if i < len(st.session_state["chat_history"]) - 2:
                st.divider()
