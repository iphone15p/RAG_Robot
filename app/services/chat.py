import uuid
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.rag import get_retriever
from app.core.config import llm
from app.services.db import (
    init_db, create_session, delete_session, get_all_sessions,
    get_session_messages, add_message, update_session_title, get_message_count
)

# 初始化数据库
init_db()

# 提示词模板
prompt = ChatPromptTemplate.from_template("""
# 角色
你是一个严谨的专业问答助手。你的任务是严格根据提供的[参考内容]解答[问题]。

# 工作规则
1. 事实锚定：你的回答必须完全基于[参考内容]中的信息。
2. 拒绝捏造：如果[参考内容]无法回答[问题]，或者信息不完整，请直接回复：“抱歉，提供的参考信息中没有相关答案。”，绝不能使用自己的内部知识进行编造或补充。
3. 结合语境：在回答时，请联系[历史对话]的上下文，但事实依旧以[参考内容]为准。
4. 输出格式：回答必须直接、精炼，去除非必要的寒暄，将长度控制在 3 句话以内。

# 数据输入
[历史对话]：
{history}

[参考内容]：
{context}

[问题]：
{question}
""")

def get_answer(question: str, session_id: str):
    """根据问题和session_id生成流式回答"""
    # 如果session不存在就新建
    sessions = get_all_sessions()
    if not any(s["id"] == session_id for s in sessions):
        session_id = create_session(
            str(uuid.uuid4())[:8],
            f"会话 {datetime.now().strftime('%H:%M')}",
            datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )

    # 检索相关文档
    retriever = get_retriever()
    docs = retriever.invoke(question)
    context = "\n".join([d.page_content for d in docs])
    sources = list(set([d.metadata.get("source", "未知") for d in docs]))

    # 取当前会话历史
    history = get_session_messages(session_id)
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    chain = prompt | llm | StrOutputParser()

    answer = ""

    def generate():
        nonlocal answer
        for chunk in chain.stream({"context": context, "question": question, "history": history_text}):
            answer += chunk
            yield chunk
        # 流式输出完毕后存入数据库
        add_message(session_id, "user", question)
        add_message(session_id, "assistant", answer)
        # 用第一个问题作为会话标题（简短版）
        if get_message_count(session_id) == 2:
            short_title = question[:12]
            if len(question) > 12:
                short_title += "..."
            update_session_title(session_id, short_title)

    return generate, sources
