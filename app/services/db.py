import sqlite3
import os

DB_PATH = "data/chat.db"

def get_db():
    """获取数据库连接"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row # 可以按字段名访问
    return conn

def init_db():
    """初始化数据库表"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = get_db()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS sessions (
            id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            sources TEXT DEFAULT '',
            FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
        );
    """)
    conn.commit()
    conn.close()

# 会话操作
def create_session(session_id: str = None, title: str = None, created_at: str = None):
    """新建会话，参数为空时自动生成"""
    import uuid
    from datetime import datetime
    if session_id is None:
        session_id = str(uuid.uuid4())[:8]
    if title is None:
        title = f"会话 {datetime.now().strftime('%H:%M')}"
    if created_at is None:
        created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = get_db()
    conn.execute("INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
                 (session_id, title, created_at))
    conn.commit()
    conn.close()
    return session_id

def delete_session(session_id: str):
    conn = get_db()
    conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
    conn.commit()
    conn.close()

def get_all_sessions():
    conn = get_db()
    rows = conn.execute("SELECT id, title, created_at FROM sessions ORDER BY created_at DESC").fetchall()
    conn.close()
    return [dict(row) for row in rows] # 数组里面是字典

def get_session_messages(session_id: str):
    conn = get_db()
    rows = conn.execute(
        "SELECT role, content, sources FROM messages WHERE session_id = ? ORDER BY id",
        (session_id,)
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def add_message(session_id: str, role: str, content: str):
    conn = get_db()
    conn.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                 (session_id, role, content))
    conn.commit()
    conn.close()

def update_session_title(session_id: str, title: str):
    conn = get_db()
    conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
    conn.commit()
    conn.close()

def get_message_count(session_id: str) -> int:
    conn = get_db()
    count = conn.execute(
        "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?", (session_id,)
    ).fetchone()["cnt"]
    conn.close()
    return count

# 新增函数
def add_message_with_sources(session_id: str, role: str, content: str, sources: str = ""):
    conn = get_db()
    conn.execute(
        "INSERT INTO messages (session_id, role, content, sources) VALUES (?, ?, ?, ?)",
        (session_id, role, content, sources)
    )
    conn.commit()
    conn.close()
