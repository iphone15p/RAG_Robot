import os
from langchain_community.document_loaders import Docx2txtLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from app.core.config import embeddings

def load_documents(folder_path="documents"):
    """读取documents文件夹里所有支持的文档"""
    docs = []
    for file in os.listdir(folder_path):
        filepath = os.path.join(folder_path, file)
        if file.endswith(".docx"):
            # 加载Word文档
            loader = Docx2txtLoader(filepath)
            docs.extend(loader.load())
        elif file.endswith(".txt"):
            # 加载txt文本文件
            loader = TextLoader(filepath, encoding="utf-8")
            docs.extend(loader.load())
    return docs

def build_vectorstore():
    """读取文档 → 切片 → 向量化 → 存入Chroma"""
    docs = load_documents()
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(chunks, embeddings, persist_directory="chroma_db")
    return vectorstore


def get_retriever():
    """加载已有向量库，返回检索器"""
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    return vectorstore.as_retriever()



def add_document(filepath: str):
    """只向量化新上传的文档，不重建整个库"""
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
    vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
    vectorstore.add_documents(chunks) # 往向量库中添加新文档
