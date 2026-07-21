# 任务四：
# 1. 用 @tool 装饰器定义一个工具函数 get_weather(city: str) -> str
# 2. 用 llm.bind_tools([get_weather]) 把工具绑定到模型
# 3. 发一条消息 "北京今天天气怎么样"
# 4. 打印返回的 AIMessage，看 .tool_calls 字段长什么样
# 5. 手动执行工具函数，把结果包成 ToolMessage 塞回消息列表
# 6. 再调一次模型，看它怎么把工具结果转成自然语言回答

import os
import json
from dotenv import load_dotenv
from langchain_core.outputs.chat_generation import ChatGenerationChunk
import requests
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage
from typing_extensions import Iterator
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.outputs import ChatGeneration, ChatResult


root_path = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(root_path, ".env")
load_dotenv(env_path)

url = os.getenv("DEEPSEEK_API_OPENAI_URL")
api_key = os.getenv("DEEPSEEK_API_KEY")
model = os.getenv("DEEPSEEK_API_MODEL")

@tool
def get_current_weather(city: str) -> str:
    """获取指定城市的实时天气信息。当用户询问天气时必须调用此工具。"""
    return f"{city} 天气晴朗，温度 25 摄氏度"

class MyChatModel(SimpleChatModel):
    
    @property
    def _llm_type(self) -> str:
        return "deepseek"
    
    def bind_tools(self, tools, **kwargs):
        openai_tools = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=openai_tools, **kwargs)

    def _convernt_message(self, messages: list[BaseMessage]) -> list[dict]:
        message_list = []
        for message in messages:
            if isinstance(message, HumanMessage):
                msg_dict = {"role": "user", "content": message.content}
            elif isinstance(message, SystemMessage):
                msg_dict = {"role": "system", "content": message.content}
            elif isinstance(message, AIMessage):
                msg_dict = {"role": "assistant", "content": message.content or ""}
                if message.tool_calls:
                    api_tool_calls = []
                    for tc in message.tool_calls:
                        api_tool_calls.append({
                            "id": tc["id"],
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc["args"])
                            }
                        })
                    msg_dict["tool_calls"] = api_tool_calls
            elif isinstance(message, ToolMessage):
                msg_dict = {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id
                }
            else:
                raise ValueError(f"Unknown message type: {type(message)}")
            
            message_list.append(msg_dict)
        return message_list

    def _call(
        self, 
        message:list[BaseMessage], 
        stop=None, 
        run_manager=None,
        **kwargs) -> str:
        message_list = self._convernt_message(message)
        data = {
            "model": model,
            "messages": message_list
        }
        if "tools" in kwargs:
            data["tools"] = kwargs["tools"]
        responce = requests.post(
            url=url+"/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=data
        )
        if not responce.ok:
            print("API 错误详情:",responce.text)
        responce.raise_for_status()
        return responce.json()["choices"][0]["message"]["content"]
    
    def _stream(
        self, 
        messages:list[BaseMessage], 
        stop=None, 
        run_manager=None,
        **kwargs) -> Iterator[ChatGenerationChunk]:
        message_list = self._convernt_message(messages)
        response = requests.post(
            url=url+"/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": message_list,
                "stream": True
            }
        )
        if not response.ok:
            print("API 错误详情:",response.text)
        response.raise_for_status()
        for line in response.iter_lines():
            if line:
                line = line.decode('utf-8')
                if line == "data: [DONE]":
                    continue
                line = line[6:]
                data = json.loads(line)
                delta = data["choices"][0]["delta"]["content"]
                chunk = ChatGenerationChunk(
                    message=AIMessage(content=delta)
                )
                yield chunk
    
    def _convert_tool_calls(self, raw_tool_calls):
        result = []
        for tc in raw_tool_calls:
            result.append({
                "id": tc["id"],
                "name": tc["function"]["name"],
                "args": json.loads(tc["function"]["arguments"])
            })
        return result
    
    def _generate(
        self, 
        messages,
        stop=None,
        run_manager=None,
        **kwarges
    ):
        message_list = self._convernt_message(messages)
        data = {
            "model": model,
            "messages": message_list
        }
        if "tools" in kwarges:
            data["tools"] = kwarges["tools"]
        
        response = requests.post(
            url=url+"/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json=data
        )
        if not response.ok:
            print("API 错误详情:",response.text)
        response.raise_for_status()
        res = response.json()["choices"][0]["message"]
        raw_tool_calls = res.get("tool_calls", [])
        tool_calls = self._convert_tool_calls(raw_tool_calls)
        ai_message = AIMessage(
            content=res.get("content", ""),
            tool_calls=tool_calls
        )
        return ChatResult(generations=[ChatGeneration(message=ai_message)])

if __name__ == "__main__":
    messages = [
        SystemMessage(content="当用户询问天气时，你必须先调用 get_current_weather 工具获取天气信息，然后根据工具返回的结果直接回答用户。"),
        HumanMessage(content="厦门今天天气怎么样")
    ]
    llm = MyChatModel()
    llm_with_tools = llm.bind_tools([get_current_weather])
    result = llm_with_tools.invoke(messages)
    print(result.tool_calls)
    tool_call = result.tool_calls[0]

    tool_result = get_current_weather.invoke(tool_call["args"])
    print("工具返回:", tool_result)

    # 把结果包成 ToolMessage
    tool_message = ToolMessage(
        content=tool_result,
        tool_call_id=tool_call["id"]
    )

    # 拼成新的消息历史
    new_messages = messages + [result, tool_message]

    final_result = llm_with_tools.invoke(new_messages)
    print(final_result.content)