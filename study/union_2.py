# 任务二：
#     1. 创建 HumanMessage，打印 .content 和 .type
#     2. 创建 SystemMessage + HumanMessage，放到一个列表里
#     3. 写一个函数 pretty_print(messages: list)，遍历打印每条消息的角色和内容
#     4. 加一条 AIMessage，看它的 type 是什么

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# 用户消息
human_message = HumanMessage(content="你好")
print(human_message.content)
print(human_message.type)

system_message = SystemMessage(content="你是一个专业的翻译")
print(system_message.content)
print(system_message.type)

messages = [system_message, human_message]

def pretty_print(messages: list):
    for message in messages:
        print(f"{message.type}: {message.content}")

aimessage = AIMessage(content="你好")
print(aimessage.content)
print(aimessage.type)


print(isinstance(human_message, HumanMessage))
print(isinstance(system_message, SystemMessage))
print(isinstance(aimessage, AIMessage))

messages = [SystemMessage("你好"), "我是一段裸字符串"]
pretty_print(messages)  # 看报什么错
