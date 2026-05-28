import os
import chromadb
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from app.core.config import embeddings

CHROMA_DIR = "chroma_db"
client = chromadb.PersistentClient(path=CHROMA_DIR)

def load_documents(folder_path="documents"):
    """读取documents文件夹里所有支持的文档"""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    docs = []
    for file in os.listdir(folder_path):
        filepath = os.path.join(folder_path, file)
        if file.endswith(".docx"):
            loader = Docx2txtLoader(filepath)
            docs.extend(loader.load())
        elif file.endswith(".txt"):
            loader = TextLoader(filepath, encoding="utf-8")
            docs.extend(loader.load())
        elif file.endswith(".pdf"):
            from langchain_community.document_loaders import PyPDFLoader
            loader = PyPDFLoader(filepath)
            docs.extend(loader.load())
    return docs

def build_vectorstore():
    """读取文档 → 切片 → 向量化 → 存入Chroma"""
    docs = load_documents()
    if not docs:
        return Chroma(client=client, embedding_function=embeddings)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, embeddings, client=client)
    return vectorstore

def get_retriever():
    """加载已有向量库，返回检索器"""
    if not os.path.exists("chroma_db/chroma.sqlite3"):
        build_vectorstore()
    vectorstore = Chroma(client=client, embedding_function=embeddings)
    return vectorstore.as_retriever()

def add_document(filepath: str):
    """只向量化新上传的文档，不重建整个库"""
    if not os.path.exists("chroma_db/chroma.sqlite3"):
        build_vectorstore()
    if filepath.endswith(".docx"):
        loader = Docx2txtLoader(filepath)
    elif filepath.endswith(".txt"):
        loader = TextLoader(filepath, encoding="utf-8")
    elif filepath.endswith(".pdf"):
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(filepath)
    else:
        return
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma(client=client, embedding_function=embeddings)
    vectorstore.add_documents(chunks)
