# 任务三：
# 1. 写一个最简单的 MyChatModel(SimpleChatModel)，只实现 _call
# 2. 实现 _llm_type 属性，返回任意字符串
# 3. 测试：llm = MyChatModel(); print(llm.invoke([HumanMessage("你好")]))
# 4. 加上 _stream 方法
# 5. 测试：for chunk in llm.stream([HumanMessage("你好")]): print(chunk.content)
# 要求：
# 1. 继承 SimpleChatModel
# 2. 实现 _llm_type（返回任意字符串即可）
# 3. 实现 _call（接收 messages，返回 str）
# 4. 用 llm.invoke([HumanMessage("你好")]) 测试
# 5. 再加 _stream 方法，用 llm.stream(...) 测试

import json
import os
from langchain_core.outputs import ChatGenerationChunk
import requests
from dotenv import load_dotenv
from typing import Iterator
from langchain_core.messages import AIMessageChunk, BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.language_models.chat_models import SimpleChatModel

root_path = os.path.dirname(os.path.dirname(__file__))
print(root_path)
env_path = os.path.join(root_path, ".env")
print(env_path)
load_dotenv(env_path)

url = os.getenv("DEEPSEEK_API_OPENAI_URL")
api_key = os.getenv("DEEPSEEK_API_KEY")
model = os.getenv("DEEPSEEK_API_MODEL")


class MyChatModel(SimpleChatModel):

    def _call(self, messages:list[BaseMessage],stop=None,run_manager=None) -> str:
        message_list = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "assistant"
            message_dit = {
                "role": role,
                "content": message.content
            }
            message_list.append(message_dit)
        response = requests.post(
            url=url+"/chat/completions",
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": model,
                "messages": message_list
            }
        )
        if not response.ok:
            print("API 错误详情:",response.text)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _convert_messages(self, messages):
        message_list = []
        for message in messages:
            if message.type == "human":
                role = "user"
            elif message.type == "system":
                role = "system"
            elif message.type == "ai":
                role = "assistant"
            message_dit = {
                "role": role,
                "content": message.content
            }
            message_list.append(message_dit)
        return message_list
    
    def _stream(self, messages: list[BaseMessage], stop=None, run_manager=None) -> Iterator[ChatGenerationChunk]:
        message_list = []
        for message in messages:
            if isinstance(message, HumanMessage):
                role = "user"
            elif isinstance(message, SystemMessage):
                role = "system"
            elif isinstance(message, AIMessage):
                role = "assistant"
            message_dit = {
                "role": role,
                "content": message.content
            }
            message_list.append(message_dit)
        responce = requests.post(
                url=url+"/chat/completions",
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": model,
                    "messages": message_list,
                    "stream": True
                }
            )
        if not responce.ok:
                print("API 错误详情:",responce.text)
        responce.raise_for_status()
        for line in responce.iter_lines():
                if line:
                    line = line.decode("utf-8")
                    if line == "data: [DONE]":
                        break
                    json_str = line[6:]
                    data = json.loads(json_str)
                    content_str = data["choices"][0]["delta"]["content"]
                    if content_str:
                        yield ChatGenerationChunk(message=AIMessageChunk(content=content_str))
    
    @property
    def _llm_type(self) -> str:
        return "my_chat_model"


if __name__ == "__main__":
    llm = MyChatModel()
    content = ""
    for chunk in llm.stream([HumanMessage("你好")]):
        print(chunk.content, end="", flush=True)

