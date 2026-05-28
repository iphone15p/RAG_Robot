import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings

# 加载.env文件里的环境变量
load_dotenv()

# 初始化LLM模型（通义千问，兼容OpenAI格式）
llm = ChatOpenAI(
    model="qwen-plus",
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 阿里云官方 Embeddings 类
embeddings = DashScopeEmbeddings(
    model="text-embedding-v4",
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),
)

# 启动时校验 API Key
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise RuntimeError(
        "未检测到 DASHSCOPE_API_KEY，请在项目根目录的 .env 文件中配置：\n"
        "DASHSCOPE_API_KEY=你的阿里云API密钥\n"
        "DASHSCOPE_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
