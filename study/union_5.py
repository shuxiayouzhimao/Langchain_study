# 任务五：
# 1. 创建一个 ChatPromptTemplate
# 2. 用 | 把 prompt | llm | StrOutputParser 串起来
# 3. 调用 chain.invoke({"topic": "Python"})
# 4. 再用 chain.stream(...) 看流式效果
# 5. 自己写一个最简单的 OutputParser（继承 BaseOutputParser，实现 parse 方法），替换掉 StrOutputParser

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import json
from dotenv import load_dotenv
from langchain_core.outputs.chat_generation import ChatGenerationChunk
import requests
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage, ToolMessage, AIMessageChunk
from typing_extensions import Iterator
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.output_parsers.base import BaseOutputParser


root_path = os.path.dirname(os.path.dirname(__file__))
env_path = os.path.join(root_path, ".env")
load_dotenv(env_path)

url = os.getenv("DEEPSEEK_API_OPENAI_URL")
api_key = os.getenv("DEEPSEEK_API_KEY")
model = os.getenv("DEEPSEEK_API_MODEL")


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
                    break
                line = line[6:]
                data = json.loads(line)
                delta = data["choices"][0]["delta"].get("content", "")
                chunk = ChatGenerationChunk(
                    message=AIMessageChunk(content=delta)
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
        **kwargs
    ):
        message_list = self._convernt_message(messages)
        data = {
            "model": model,
            "messages": message_list
        }
        if "tools" in kwargs:
            data["tools"] = kwargs["tools"]
        
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

class UpperCaseParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        return text.upper()

if __name__ == "__main__":
    print("Hello, World!")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "你是一个专业的{domain}专家，回答用英语。"),
        ("human", "请用一句话解释什么是{topic}")
    ])

    llm = MyChatModel()
    parser = StrOutputParser()
    chain = prompt | llm | UpperCaseParser()
    result = chain.invoke({
        "domain": "计算机科学",
        "topic": "人工智能"
    })
    print(result)

    for text in chain.stream({
        "domain": "计算机科学",
        "topic": "人工智能"
    }):
        print(text, end="",flush=True)
