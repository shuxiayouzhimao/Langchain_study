import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from typing import Annotated
from langchain_core.tools import tool
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from union_4 import MyChatModel, get_current_weather
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

@tool
def get_time(city: str) -> str:
    """获取指定城市的当前时间"""
    return f"{city} 的当前时间是 15:00"

llm = MyChatModel()
llm_with_tools = llm.bind_tools([get_current_weather, get_time])

def call_model(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return "__end__"

tool_by_name = {
    "get_current_weather": get_current_weather,
    "get_time": get_time
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


if __name__ == "__main__":
    graph_builder = StateGraph(AgentState)

    graph_builder.add_node("call_model", call_model)
    graph_builder.add_node("tools", tool_nodes)

    graph_builder.add_edge(START, "call_model")
    graph_builder.add_conditional_edges("call_model", should_continue)
    graph_builder.add_edge("tools", "call_model")

    graph = graph_builder.compile(checkpointer=MemorySaver())

    config_default = {"configurable": {"thread_id": "default"}}

    result = graph.invoke({
        "messages": [
            HumanMessage(content="厦门天气和纽约时间")
        ]
    }, config=config_default)
    print(result["messages"][-1].content)

    config = {"configurable": {"thread_id": "1"}}

    result1 = graph.invoke({
        "messages": [
            HumanMessage(content="我叫小明，请记住我的名字")
        ]
    }, config=config)
    print(result1["messages"][-1].content)

    result2 = graph.invoke({
        "messages": [
            HumanMessage(content="你好，我叫什么？")
        ]
    }, config=config)
    print(result2["messages"][-1].content)

    config_other = {"configurable": {"thread_id": "2"}}
    result3 = graph.invoke({
        "messages": [
            HumanMessage(content="你好，我叫什么？")
        ]
    }, config=config_other)
    print(result3["messages"][-1].content)

