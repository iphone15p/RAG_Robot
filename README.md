# RAG 知识库问答机器人

基于 RAG（检索增强生成）技术的智能问答系统，用户上传文档后即可针对文档内容进行提问。

## 技术栈

- **LLM**：阿里云 DashScope 千问（qwen-plus）
- **Embeddings**：阿里云 DashScope text-embedding-v4
- **向量数据库**：ChromaDB
- **框架**：LangChain + Streamlit + FastAPI
- **数据库**：SQLite

## 功能

- 支持 TXT / DOCX / PDF 文档上传，自动向量化入库
- 基于文档内容的智能问答，回答附带来源文档名
- 区分闲聊与知识库查询，无相关知识时明确告知
- 多会话管理，历史记录持久化
- 流式输出回复
- 双界面：Streamlit Web UI + FastAPI REST API

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

在项目根目录创建 `.env` 文件，填入：

```env
DASHSCOPE_API_KEY=你的阿里云API密钥
DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

### 3. 放入文档

将要检索的 TXT / DOCX / PDF 文件放入 `documents/` 文件夹。

### 4. 启动

**Streamlit UI（推荐）：**

```bash
streamlit run ui.py
```

**FastAPI 服务：**

```bash
uvicorn app.main:app
```

启动后浏览器访问 `http://localhost:8501`（Streamlit）或 `http://localhost:8000`（FastAPI）。

## 项目结构

```
├── app/
│   ├── main.py              # FastAPI 入口
│   ├── core/config.py       # LLM 和 Embeddings 初始化
│   └── services/
│       ├── chat.py          # RAG 问答核心逻辑
│       ├── rag.py           # 文档加载、切片、向量化
│       └── db.py            # SQLite 数据库操作
├── ui.py                    # Streamlit 界面
├── documents/               # 文档存放目录
├── data/                    # 会话数据库（自动生成）
├── chroma_db/               # 向量库（自动生成）
└── requirements.txt
```
