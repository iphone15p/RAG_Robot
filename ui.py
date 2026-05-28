import streamlit as st
import os, shutil
from app.services.chat import create_session, get_answer
from app.services.db import delete_session, get_all_sessions, get_session_messages
from app.services.rag import build_vectorstore, add_document

# ========== 初始化 ==========

if not os.path.exists("chroma_db/chroma.sqlite3"):
    build_vectorstore()

if "current_session_id" not in st.session_state:
    sessions_data = get_all_sessions()
    if sessions_data:
        st.session_state.current_session_id = sessions_data[0]["id"]
    else:
        st.session_state.current_session_id = None

# ========== 侧边栏 ==========

with st.sidebar:
    st.title("💬 会话管理")

    if st.button("➕ 新建会话", use_container_width=True):
        st.session_state.current_session_id = create_session()
        st.rerun()

    st.divider()

    sessions_list = get_all_sessions()

    if not sessions_list:
        st.caption("暂无会话")
    else:
        for s in sessions_list:
            col1, col2 = st.columns([4, 1])
            with col1:
                if st.button(
                    f"{'▶ ' if s['id'] == st.session_state.current_session_id else ''}{s['title']}  {s['created_at']}",
                    key=s["id"], use_container_width=True
                ):
                    st.session_state.current_session_id = s["id"]
                    st.rerun()
            with col2:
                if st.button("❌", key=f"del_{s['id']}"):
                    delete_session(s["id"])
                    if s["id"] == st.session_state.current_session_id:
                        st.session_state.current_session_id = None
                    st.rerun()

    st.divider()

    st.subheader("📁 上传文档")
    uploaded_file = st.file_uploader("上传 PDF / TXT / DOCX", type=["txt", "docx", "pdf"])
    if uploaded_file:
        save_path = f"documents/{uploaded_file.name}"
        with open(save_path, "wb") as f: # wb 二进制写入
            shutil.copyfileobj(uploaded_file, f)
        add_document(save_path) # 为了放向量库给ai知道
        st.success(f"{uploaded_file.name} 上传成功")

# ========== 主界面 ==========

st.title("📚 知识库问答机器人")
st.caption("基于用户上传和原有知识智能问答")

if st.session_state.current_session_id:
    messages = get_session_messages(st.session_state.current_session_id)
    # ui.py 历史消息渲染改为：
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):  # 有来源才显示
                st.caption("📄 来源：" + msg["sources"])

if question := st.chat_input("请输入你的问题..."):
    if not st.session_state.current_session_id:
        st.session_state.current_session_id = create_session()

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("思考中..."):
            generate, sources = get_answer(question, st.session_state.current_session_id)
            placeholder = st.empty()
            answer = ""
            for chunk in generate(): # generate 模型返回内容
                answer += chunk
                placeholder.write(answer)
            if sources:
                st.caption("📄 来源：" + "、".join(sources))  # sources 参考文件名

    st.rerun()
