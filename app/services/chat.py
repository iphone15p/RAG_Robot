import os
import uuid
from datetime import datetime
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.services.rag import get_retriever
from app.core.config import llm
from app.services.db import (
    init_db, create_session, delete_session, get_all_sessions,
    get_session_messages, add_message, update_session_title, get_message_count, add_message_with_sources,
)
# ... existing code ...


# 初始化数据库
init_db()

# 提示词模板
prompt = ChatPromptTemplate.from_template("""
# 角色
你是一个专业问答助手。

# 工作规则
1. 如果用户问题是闲聊、打招呼、问候等与知识库无关的内容，直接友好回复，回复开头加上标记 [CHAT]。
2. 如果问题与知识库相关，你的回答必须完全基于[参考内容]中的信息，回复开头加上标记 [RAG]。
3. 拒绝捏造：如果[参考内容]无法回答[问题]，回复开头加上标记 [RAG]，然后回复："抱歉，提供的参考信息中没有相关答案。"
4. 结合语境：在回答时，请联系[历史对话]的上下文。
5. 输出格式：回答必须直接、精炼，去除非必要寒暄，控制在 3 句话以内。标记不要显示给用户。

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
    # 去掉路径前缀，只保留文件名
    sources = [os.path.basename(s) for s in sources]
    sources_str = "、".join(sources)

    # 取当前会话历史
    history = get_session_messages(session_id)
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])

    chain = prompt | llm | StrOutputParser()
    answer = ""

    def generate():
        nonlocal answer
        buffer = ""
        tag_processed = False  # 核心标志位：是否已经处理完开头的标签

        for chunk in chain.stream({"context": context, "question": question, "history": history_text}):
            answer += chunk

            if not tag_processed:
                buffer += chunk
                # 攒够字符后，检查标签是否已经完整出现在 buffer 中
                if "[CHAT]" in buffer or "[RAG]" in buffer:
                    # 替换掉标签，并去除标签后可能跟着的空格或换行 (.lstrip)
                    buffer = buffer.replace("[CHAT]", "").replace("[RAG]", "").lstrip()
                    tag_processed = True
                    # 如果过滤后 buffer 里还有实际内容，就发送给前端
                    if buffer:
                        yield buffer
                        buffer = ""
                # 防御性编程：如果前 10 个字符都没有出现标签，说明大模型没有按指令输出，直接放弃拦截
                elif len(buffer) > 10:
                    tag_processed = True
                    yield buffer
                    buffer = ""
            else:
                # 标签拦截期已过，直接顺畅地输出后续的 chunk
                yield chunk

        # --- 生成结束后，处理数据库保存逻辑 ---

        # 判断是否使用了知识库 (包含 [RAG] 标记)
        is_rag = "[RAG]" in answer

        # 清理最终要保存到数据库的纯净文本
        final_answer = answer.replace("[CHAT]", "").replace("[RAG]", "").strip()

        # 💡 逻辑优化：如果模型回复没有找到答案，即使是 RAG 模式，也不应该保存/显示来源文档
        if is_rag and "抱歉，提供的参考信息中没有相关答案" not in final_answer:
            final_sources = sources_str
        else:
            final_sources = ""

        add_message_with_sources(session_id, "user", question, "")
        add_message_with_sources(session_id, "assistant", final_answer, final_sources)

        if get_message_count(session_id) == 2:
            short_title = question[:12]
            if len(question) > 12:
                short_title += "..."
            update_session_title(session_id, short_title)

    return generate, sources