import os
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

_retriever = None  # 惰性初始化用

def _init_rag_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if os.path.exists("study/chroma_db"):
        vectorstore = Chroma(
            persist_directory="study/chroma_db",
            embedding_function=embeddings
        )
    else:
        with open("study/knowledge.md", "r", encoding="utf-8") as f:
            text = f.read()
        documents = [Document(page_content=text)]
        splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
        chunks = splitter.split_documents(documents)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="study/chroma_db"
        )

    return vectorstore.as_retriever(search_kwargs={"k": 2})

def _get_retriever():
    global _retriever
    if _retriever is None:
        _retriever = _init_rag_retriever()
    return _retriever

@tool
def get_current_weather(city: str) -> str:
    """获取指定城市的实时天气信息"""
    return f"{city} 天气晴朗，温度 25 摄氏度"

@tool
def get_time(city: str) -> str:
    """获取指定城市的当前时间"""
    return f"{city} 的当前时间是 15:00"

@tool
def query_knowledge_base(question: str) -> str:
    """查询公司内部知识库"""
    retriever = _get_retriever()
    docs = retriever.invoke(question)
    return "\n\n".join([doc.page_content for doc in docs])
