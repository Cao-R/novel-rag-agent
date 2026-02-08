import streamlit as st
import requests
import json

# 设置页面标题
st.set_page_config(page_title="多智能体小说助手", page_icon="")
st.title("本地小说 RAG + 联网搜索助手")

# 初始化聊天历史
if "messages" not in st.session_state:
    st.session_state.messages = []

# 显示聊天历史
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 处理用户输入
if prompt := st.chat_input("请输入关于小说的问题..."):
    # 1. 显示用户问题
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. 调用后端 API 并显示流式结果
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        # 连接你的 FastAPI 服务
        # 注意：这里调用的是 GET /chat/stream 接口
        api_url = f"http://127.0.0.1:8000/chat/stream?query={prompt}"
        
        try:
            with requests.get(api_url, stream=True) as r:
                if r.status_code == 200:
                    # 处理 SSE 流式响应
                    for line in r.iter_lines():
                        if line:
                            decoded_line = line.decode('utf-8')
                            if decoded_line.startswith("data: "):
                                # 去掉 'data: ' 前缀
                                content = decoded_line[6:]
                                if content == "[DONE]":
                                    break
                                # 拼接内容并刷新显示
                                full_response += content
                                message_placeholder.markdown(full_response + "▌")
                                
                    message_placeholder.markdown(full_response)
                else:
                    st.error(f"服务器错误: {r.status_code}")
        except Exception as e:
            st.error(f"连接失败，请确认 server.py 是否正在运行。\n错误信息: {e}")

    # 3. 保存助手回答到历史
    st.session_state.messages.append({"role": "assistant", "content": full_response})