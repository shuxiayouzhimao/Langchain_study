"""XlingAI Agent 核心逻辑。

从 study/union_8.py 的成果抽出来，供 app.main 调用。
模块级别导出 `graph` 和 `memory_store`。
"""

import os
import sys
import json
from datetime import datetime

# 让 app 目录能找到同级的 study 目录
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from typing import Annotated
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver

from study.union_4 import MyChatModel
from study.study_tools import get_current_weather, get_time, query_knowledge_base


# ========== LLM 初始化 ==========
llm = MyChatModel()
llm_with_tools = llm.bind_tools([get_current_weather, get_time, query_knowledge_base])


# ========== 长期记忆 ==========
class LongTermMemory:
    def __init__(self, memory_file="study/memory.json"):
        self.memory_file = memory_file
        self.facts = self._load()
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

    def _cosine_similarity(self, vec1, vec2):
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

    def _find_similar_fact(self, content: str, threshold: float = 0.85):
        if not self.facts:
            return None
        new_vec = self.embeddings.embed_query(content)
        for fact in self.facts:
            old_vec = self.embeddings.embed_query(fact["content"])
            similarity = self._cosine_similarity(new_vec, old_vec)
            if similarity > threshold:
                return fact
        return None

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
            "category": category,
            "created_at": datetime.now().isoformat(),
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


memory_store = LongTermMemory()


# ========== 反思函数 ==========
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

    conversation_text = "\n".join([f"{m.type}: {m.content}" for m in messages])
    result = llm.invoke(prompt.invoke({"conversation": conversation_text}))

    try:
        content = result.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        facts = json.loads(content.strip())
        return facts if isinstance(facts, list) else []
    except json.JSONDecodeError:
        return []


# ========== LangGraph 状态与节点 ==========
class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def call_model(state: AgentState):
    messages = state["messages"]

    # 注入长期记忆
    facts = memory_store.get_facts()
    memory_text = "\n".join([f["content"] for f in facts])

    system_msg = SystemMessage(
        content=f"""
        你是 XlingAI 个人助手。
        你有能力查询公司内部知识库、获取天气信息和时间。
        以下是你已经记住的用户信息：
        {memory_text}
        请根据以上信息和当前对话回答问题。
        """
    )
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
        tool_name = tool_call["name"]
        tool = tool_by_name.get(tool_name)
        if tool is None:
            tool_result = f"未知工具: {tool_name}"
        else:
            tool_result = tool.invoke(tool_call["args"])

        tool_messages.append(
            ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
        )

    return {"messages": tool_messages}


def reflect_node(state: AgentState):
    messages = state["messages"]
    facts = reflect_on_conversation(messages, llm)
    for fact in facts:
        memory_store.add_fact(fact, category="user_info")
    print(f"[反思] 提取到 {len(facts)} 条记忆：{facts}")
    return {"messages": []}


# ========== 构建图（模块级别，只执行一次） ==========
_graph_builder = StateGraph(AgentState)

_graph_builder.add_node("call_model", call_model)
_graph_builder.add_node("tools", tool_nodes)
_graph_builder.add_node("reflect", reflect_node)

_graph_builder.add_edge(START, "call_model")
_graph_builder.add_conditional_edges(
    "call_model",
    should_continue,
    {"tools": "tools", "reflect": "reflect"},
)
_graph_builder.add_edge("tools", "call_model")
_graph_builder.add_edge("reflect", END)

# 用 MemorySaver 实现同一 thread_id 内的短期记忆
graph = _graph_builder.compile(checkpointer=MemorySaver())
