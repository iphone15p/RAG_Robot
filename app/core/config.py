import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

# 加载.env文件里的环境变量
load_dotenv()

# 获取 API Key：优先环境变量，Streamlit Cloud 则从 secrets 读取
def _get_api_key():
    key = os.getenv("DASHSCOPE_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        key = st.secrets.get("DASHSCOPE_API_KEY", "")
    except Exception:
        pass
    return key

DASHSCOPE_API_KEY = _get_api_key()

# 启动时校验 API Key
if not DASHSCOPE_API_KEY:
    raise RuntimeError(
        "未检测到 DASHSCOPE_API_KEY，请在项目根目录的 .env 文件中配置：\n"
        "DASHSCOPE_API_KEY=你的阿里云API密钥\n"
        "DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1\n\n"
        "Streamlit Cloud 用户请在 App Settings → Secrets 中添加：\n"
        "DASHSCOPE_API_KEY = \"你的阿里云API密钥\""
    )

# 初始化LLM模型（通义千问，兼容OpenAI格式）
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=DASHSCOPE_API_KEY,
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 阿里云官方 Embeddings 类
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=DASHSCOPE_API_KEY,
)
