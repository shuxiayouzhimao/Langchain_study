# 任务一：用纯 requests 调用 DeepSeek API
# 1. 从 .env 读配置
# 2. 手动拼 messages 列表（字典格式，不是 LangChain 对象）
# 3. 发 POST 请求到 /chat/completions
# 4. 打印返回的 content
# 5. 加上 stream=True，逐行打印流式响应
import json
import os
import sys
import requests
from dotenv import load_dotenv

# 项目根目录地址
project_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
env_path = os.path.join(project_path, ".env")
print(env_path)

# 从.env读取配置
load_dotenv(dotenv_path=env_path)

url = os.getenv("DEEPSEEK_API_OPENAI_URL")
print(url)

api_key = os.getenv("DEEPSEEK_API_KEY")
print(api_key)

model = os.getenv("DEEPSEEK_API_MODEL")
print(model)



# 拼message列表
messages = [
    {"role": "system", "content":"you are a professional translator"},
    {"role": "user", "content":"你好"}
]

# 发 POST 请求到 /chat/completions

responce = requests.post(
    url=url+"/chat/completions",
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json = {
        "model": model,
        "messages": messages,
        "stream": True
    }
)

responce.raise_for_status()
all_content = ""
for line in responce.iter_lines():
    if line:
        line = line.decode("utf-8")

        if line == "data: [DONE]":
            break

        # 去掉 “data：”前缀
        json_str = line.replace("data: ", "")
        data = json.loads(json_str)
        content_data = data['choices'][0]['delta']['content']
        all_content += content_data
print(all_content)
