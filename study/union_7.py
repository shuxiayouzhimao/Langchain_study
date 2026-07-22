import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from study.union_4 import MyChatModel

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate

# 1. 加载文档（用 Python 内置读取替代 TextLoader）
with open("study/knowledge.md", "r", encoding="utf-8") as f:
    text = f.read()
documents = [Document(page_content=text, metadata={"source": "study/knowledge.md"})]

# 2. 切片
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50
)
chunks = splitter.split_documents(documents)

print(f"文档共 {len(documents)} 页，分 {len(chunks)} 个块")
for i, chunk in enumerate(chunks):
    print(f"---片段{i}---")
    print(chunk.page_content)

# 3. 生成 Embedding 并存入 Chroma
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="study/chroma_db"
)
print("向量数据库已创建并持久化")

# 4. 检索
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

question = "员工年假有多少天？"
docs = retriever.invoke(question)

print(f"\n问题: {question}")
for i, doc in enumerate(docs):
    print(f"---相关文档{i}---")
    print(doc.page_content)

# 5. 调用 LLM 生成答案
llm = MyChatModel()

prompt = ChatPromptTemplate.from_template("""
你是一个专业的员工服务助手，你的任务是回答员工关于公司政策的问题。
请根据提供的相关文档内容，回答员工的问题。如果相关文档不包含足够的信息，
请回答“根据已知信息无法回答该问题”。

相关文档:
{context}

问题: {question}
""")

context = "\n\n".join([doc.page_content for doc in docs])
messages = prompt.invoke({
    "context": context,
    "question": question
})
response = llm.invoke(messages)
print(f"\n最终回答:\n{response.content}")
