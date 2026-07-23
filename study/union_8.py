import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import os
import json
from datetime import datetime
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from union_4 import MyChatModel
from typing import Annotated
from langgraph.graph import add_messages
from typing_extensions import TypedDict
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from study_tools import get_current_weather, get_time, query_knowledge_base

llm = MyChatModel()
llm_with_tools = llm.bind_tools([get_current_weather, get_time, query_knowledge_base])

class LongTermMemory:
    def __init__(self, memory_file="study/memory.json"):
        self.memory_file = memory_file
        self.facts = self._load()
        self.embeddings = HuggingFaceEmbeddings(
            model_name = "sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def _cosine_similarity(self, vec1, vec2):
        """计算两个向量的余弦相似度"""
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    
    def _find_similar_fact(self, content: str, threshold: float = 0.85):
        """查找是否有语义相似的记忆"""
        if not self.facts:
            return None
        
        new_vec = self.embeddings.embed_query(content)

        for fact in self.facts:
            old_vec = self.embeddings.embed_query(fact["content"])
            similarity = self._cosine_similarity(new_vec, old_vec)
            if similarity > threshold:
                return fact

    def _load(self):
        if os.path.exists(self.memory_file):
            with open(self.memory_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return []
    
    def add_fact(self, content: str, category: str = "general"):
        similar_fact = self._find_similar_fact(content)
        if similar_fact:
            print(f"[记忆去重] 跳过重复内容：{content}")
            return
        fact = {
            "id": len(self.facts) + 1,
            "content": content,
            "category": category, # 比如 preference, fact, summary
            "created_at": datetime.now().isoformat()
        }
        self.facts.append(fact)
        self._save()
    
    def get_facts(self, category=None) -> list:
        if category:
            return [f for f in self.facts if f["category"] == category]
        return self.facts
    
    def _save(self):
        with open(self.memory_file, "w", encoding="utf-8") as f:
            json.dump(self.facts, f, ensure_ascii=False, indent=2)

def reflect_on_conversation(messages: list, llm) -> list:
    prompt = ChatPromptTemplate.from_template(
        """
        请从以下对话中提取值得长期记住的关键事实。
        只提取关于用户偏好、习惯、重要信息的事实，不要提取闲聊内容。
        每个事实用一句话描述。

        对话：
        {conversation}

        请严格输出 JSON 数组格式，不要加任何解释：
        ["事实1", "事实2", ...]

        如果没有值得记住的事实，输出空数组：[]
        """
    )

    # 把消息列表转成文本
    conversation_text = "\n".join([
        f"{m.type}: {m.content}" for m in messages
    ])

    result = llm.invoke(prompt.invoke({
        "conversation": conversation_text
    }))

    # 解析 JSON 数组
    try:
        content = result.content.strip()
        # 如果模型返回 “”“json...”“”代码块，去掉它
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        facts = json.loads(content.strip())
        return facts if isinstance(facts, list) else []
    except json.JSONDecodeError:
        return []

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def call_model(state: AgentState):
    messages = state["messages"]
    
    # 读取长期记忆
    facts = memory_store.get_facts()
    memory_text = "\n".join([f["content"] for f in facts])

    # 构造带记忆的 system prompt
    system_msg = SystemMessage(
        content=f"""
        你是一个智能助手，有能力回答用户问题并执行相关操作。
        你有能力查询公司内部知识库、获取天气信息和时间。
        以下是你已经记住的用户信息：
        {memory_text}
        请根据以上信息和当前对话回答问题。
        """
    )
    # 把 system prompt 放在最前面

    full_messages = [system_msg] + messages
    response = llm_with_tools.invoke(full_messages)

    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "reflect"

tool_by_name = {
    "get_current_weather": get_current_weather,
    "get_time": get_time,
    "query_knowledge_base": query_knowledge_base,
}

def tool_nodes(state: AgentState) -> dict:
    last_message = state["messages"][-1]
    tool_messages = []

    for tool_call in last_message.tool_calls:
        # 根据name执行对应工具
        tool_name = tool_call["name"]
        tool = tool_by_name.get(tool_name)
        if tool is None:
            tool_result = f"未知工具: {tool_name}"
        else:
            tool_result = tool.invoke(tool_call["args"])
        
        tool_messages.append(
            ToolMessage(
                content=tool_result,
                tool_call_id=tool_call["id"]
            )
        )
    
    return {"messages": tool_messages}

memory_store = LongTermMemory()

def reflect_node(state: AgentState):
    messages = state["messages"]
    facts = reflect_on_conversation(messages, llm)
    for fact in facts:
        memory_store.add_fact(fact, category="user_info")
    print(f"[反思] 提取到 {len(facts)} 条记忆：{facts}")
    return {"messages": []}   # 不添加新消息，不改变对话

if __name__ == "__main__":
    # 1. 测试反思函数
    # messages = [
    #     HumanMessage(content="我叫小明，我喜欢吃火锅"),
    #     AIMessage(content="好的，我记住了。")
    # ]
    # facts = reflect_on_conversation(messages, llm)
    # print("反思结果：", facts)

    # 2. 组装并运行图
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("call_model", call_model)
    graph_builder.add_node("tools", tool_nodes)
    graph_builder.add_node("reflect", reflect_node)
    
    graph_builder.add_edge(START, "call_model")
    graph_builder.add_conditional_edges(
        "call_model",
        should_continue,
        {
            "tools": "tools",
            "reflect": "reflect"
        }
    )
    graph_builder.add_edge("tools", "call_model")
    graph_builder.add_edge("reflect", END)

    graph = graph_builder.compile()
    
    # 第一轮： 告诉助手偏好
    result_1 = graph.invoke({
        "messages": [
            HumanMessage(content="我是小明，我喜欢吃火锅")
        ]
        }, 
        config={"configurable": {"thread_id": "session_1"}}
    )
    print("第一次结果：", result_1["messages"][-1].content)
    # 第二轮：询问偏好
    print("\n=== 第二轮 ===")
    result_2 = graph.invoke({
        "messages": [
            HumanMessage(content="我叫什么？喜欢吃什么？")
        ]
        }, 
        config={"configurable": {"thread_id": "session_2"}}
    )
    print("第二次结果：", result_2["messages"][-1].content)

    # 第三轮：测试知识库工具
    print("\n=== 第三轮：知识库 ===")
    result_3 = graph.invoke({
        "messages": [
            SystemMessage(content="你可以查询天气、时间，也可以查询公司内部知识库回答政策问题。"),
            HumanMessage(content="员工年假有多少天？")
        ]
        },
        config={"configurable": {"thread_id": "session_3"}}
    )
    print("第三轮结果：", result_3["messages"][-1].content)
