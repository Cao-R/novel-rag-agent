import streamlit as st
from langchain_core.messages import HumanMessage
# 导入main_agent.py 里构建好的 app 对象
from main_agent import app 

st.title("本地小说 RAG 助手")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示历史消息
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 接收用户输入
if prompt := st.chat_input("问问关于小说的问题..."):
    # 显示用户消息
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 调用 Agent
    with st.chat_message("assistant"):
        with st.spinner("正在查阅古籍..."):
            initial_state = {"messages": [HumanMessage(content=prompt)]}
            result = app.invoke(initial_state)
            response = result["messages"][-1].content
            st.markdown(response)
    
    # 保存助手消息
    st.session_state.messages.append({"role": "assistant", "content": response})