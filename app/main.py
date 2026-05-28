from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from app.services.rag import build_vectorstore
from app.services.chat import get_answer, create_session, delete_session, get_all_sessions, get_session_messages
import shutil, urllib.parse

app = FastAPI()

# 启动时建向量库
build_vectorstore()

class Question(BaseModel):
    question: str
    session_id: str  # 每次请求带上session_id

@app.post("/chat")
def chat(q: Question):
    generate, sources = get_answer(q.question, q.session_id)
    encoded_sources = ",".join([urllib.parse.quote(s) for s in sources])
    return StreamingResponse(generate(), media_type="text/plain", headers={"X-Sources": encoded_sources})

@app.post("/session/new")
def new_session():
    """新建会话"""
    session_id = create_session()
    return {"session_id": session_id}

@app.delete("/session/{session_id}")
def remove_session(session_id: str):
    """删除会话"""
    delete_session(session_id)
    return {"message": "删除成功"}

@app.get("/sessions")
def list_sessions():
    """获取所有会话列表"""
    return {"sessions": get_all_sessions()}

@app.get("/session/{session_id}/messages")
def session_messages(session_id: str):
    """获取指定会话的消息历史"""
    return {"messages": get_session_messages(session_id)}

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    save_path = f"documents/{file.filename}"
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    build_vectorstore()
    return {"message": f"{file.filename} 上传成功"}