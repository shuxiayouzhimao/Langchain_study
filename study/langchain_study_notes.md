# LangChain 学习笔记 - 5 单元

> 基于 DeepSeek API 的 5 单元实战练习总结。
> 本文档同时作为学习者复习资料 和 后续 AI 助手的上下文输入。

---

## 📌 学习者画像（给 AI 看）

- **学习背景**：非科班，代码主要由 AI 辅助完成，自己通过手敲重写来理解。
- **当前水平**：已完成 LangChain Core 基础 5 单元，能独立写出自定义 ChatModel、工具调用、LCEL 链。
- **理解深度**：知道代码"怎么写"，但对框架底层设计模式、类型系统、错误排查还在熟悉中。
- **学习方式**：偏好任务驱动 + 手打代码 + 报错后由 AI 引导排查，不喜欢直接复制答案。
- **项目目标**：构建一个**个人 AI 助手**，具备长期记忆、知识库、自动学习能力，能在日常生活中提供帮助。
- **当前瓶颈**：
  - 自定义组件与框架接口之间的格式转换容易出错。
  - 对 LangGraph、RAG、持久化记忆等高级主题尚未接触。
  - 需要自己把学到的组件拼装成完整应用。

### 已有文件索引

| 文件 | 内容 | 状态 |
|------|------|------|
| [`study/union_1.py`](union_1.py) | 纯 requests 调用 DeepSeek API（流式解析） | ✅ 完成 |
| [`study/union_2.py`](union_2.py) | LangChain 消息对象体系 | ✅ 完成 |
| [`study/union_3.py`](union_3.py) | 手写 MyChatModel（继承 SimpleChatModel） | ✅ 完成 |
| [`study/union_4.py`](union_4.py) | Tool Calling 完整实现 | ✅ 完成 |
| [`study/union_5.py`](union_5.py) | LCEL 管道与自定义 Parser | ✅ 完成 |
| [`study/union_6.py`](union_6.py) | LangGraph 状态图 + 多工具 + 记忆 | ✅ 完成 |

### 给 AI 的协作提示

- 解释新概念时，优先用**具体代码示例**和**类比**。
- 遇到报错，先引导用户检查输入输出格式、类型匹配、环境变量。
- 建议把复杂任务拆成 3-5 个小步骤，每步有明确的验收标准。
- 用户愿意手敲代码，不要直接给出完整答案，而是给出框架让其填充。
- 用户的目标是个人 AI 助手，后续学习建议围绕 **LangGraph → RAG → 持久化记忆 → 部署** 展开。

---

## 单元 1：消息对象体系

### 核心概念

LangChain 中所有消息都继承自 `BaseMessage`。不同的消息类型代表对话中不同的角色。

| 消息类 | `.type` 返回值 | API role | 说明 |
|--------|----------------|----------|------|
| `HumanMessage` | `human` | `user` | 用户说的话 |
| `AIMessage` | `ai` | `assistant` | AI 的回复 |
| `SystemMessage` | `system` | `system` | 系统提示，设定助手行为 |
| `ToolMessage` | `tool` | `tool` | 工具执行结果 |

### 关键理解

- `.type` 是 LangChain 内部命名，与 API 要求的 `role` 不完全相同。
- `BaseMessage` 是抽象父类，方法签名中通常写 `List[BaseMessage]`，不关心具体子类。
- 多态是 LangChain 设计的基础：调用方只关心"这是一条消息"，不关心具体类型。

### 代码示例

```python
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage

human = HumanMessage(content="你好")
print(human.type)      # human
print(human.content)   # 你好
print(isinstance(human, BaseMessage))  # True
```

---

## 单元 2：直接调用 API

### 核心目标

理解 LangChain 封装之下，底层到底发生了什么。

### 非流式调用

```python
import requests

response = requests.post(
    url="https://api.deepseek.com/chat/completions",
    headers={
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    },
    json={
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": "你是助手"},
            {"role": "user", "content": "你好"}
        ]
    }
)
content = response.json()["choices"][0]["message"]["content"]
```

### 流式调用

```python
response = requests.post(
    url="...",
    headers={...},
    json={..., "stream": True},
    stream=True
)

for line in response.iter_lines():
    if not line:
        continue
    text = line.decode("utf-8")
    if text == "data: [DONE]":
        break
    json_str = text[6:]  # 去掉 "data: " 前缀
    data = json.loads(json_str)
    delta = data["choices"][0]["delta"].get("content", "")
    print(delta, end="", flush=True)
```

### 关键要点

| 概念 | 说明 |
|------|------|
| `iter_lines()` | 返回生成器，需要 `for` 循环遍历 |
| `flush=True` | 强制立即输出，实现逐字打印 |
| `data: [DONE]` | SSE 流结束标志 |
| 第一个 chunk | 通常只包含 `role: assistant`，`content` 为空 |
| 中文乱码 | 因为 `bytes` 没解码，需要 `.decode("utf-8")` |

---

## 单元 3：自定义 ChatModel

### 核心目标

理解 LangChain 的 ChatModel 抽象。通过继承 `SimpleChatModel`，自己实现一个 LLM 封装。

### 模板方法模式

```
用户调用 invoke() / stream()
        ↓
框架调用 _call() / _stream()
        ↓
你写的代码调用 API
        ↓
返回结果给框架
        ↓
框架包装成 AIMessage / AIMessageChunk 返回给用户
```

### 必须实现的方法

| 方法 | 返回值 | 作用 |
|------|--------|------|
| `_call(messages, stop, run_manager, **kwargs)` | `str` | 非流式生成 |
| `_stream(messages, stop, run_manager, **kwargs)` | `Iterator[ChatGenerationChunk]` | 流式生成 |
| `_llm_type` | `str` | 标识模型类型 |
| `_generate(messages, stop, run_manager, **kwargs)` | `ChatResult` | 需要 tool_calls 时重写 |

### 关键代码

```python
from langchain_core.language_models.chat_models import SimpleChatModel
from langchain_core.messages import AIMessageChunk
from langchain_core.outputs import ChatGenerationChunk
from typing import Iterator

class MyChatModel(SimpleChatModel):
    @property
    def _llm_type(self) -> str:
        return "my-model"

    def _call(self, messages, stop=None, run_manager=None, **kwargs) -> str:
        # 调用 API，返回字符串
        ...
        return response.json()["choices"][0]["message"]["content"]

    def _stream(self, messages, stop=None, run_manager=None, **kwargs) -> Iterator[ChatGenerationChunk]:
        # 调用 API，逐块 yield
        ...
        yield ChatGenerationChunk(message=AIMessageChunk(content=chunk_text))
```

### 重要区别

| 调用方式 | 框架调用 | 返回值类型 |
|----------|----------|-----------|
| `llm.invoke(messages)` | `_call()` | `AIMessage` |
| `llm.stream(messages)` | `_stream()` | `Iterator[AIMessageChunk]` |
| `llm._call(messages)` | 直接调用私有方法 | `str` |
| `llm._stream(messages)` | 直接调用私有方法 | `Iterator[ChatGenerationChunk]` |

---

## 单元 4：Tool Calling（工具调用）

### 核心目标

让模型能够"决定"调用外部函数，并根据函数结果生成回答。

### 工具调用循环

```
用户提问
    ↓
模型返回 tool_calls（我要调工具）
    ↓
你执行对应的 Python 函数
    ↓
把结果包成 ToolMessage
    ↓
[用户消息, AI工具请求, 工具结果] 再次传给模型
    ↓
模型生成自然语言回答
```

### 关键步骤

#### 1. 定义工具

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """查询指定城市的天气"""
    return f"{city} 天气晴朗，25°C"
```

#### 2. 绑定工具

`SimpleChatModel` 默认没有实现 `bind_tools()`，需要自己在子类中实现：

```python
from langchain_core.utils.function_calling import convert_to_openai_tool

def bind_tools(self, tools, **kwargs):
    openai_tools = [convert_to_openai_tool(t) for t in tools]
    return self.bind(tools=openai_tools, **kwargs)
```

#### 3. 在 `_generate` 中透传 tools

```python
def _generate(self, messages, stop=None, run_manager=None, **kwargs):
    data = {"model": model, "messages": messages}
    if "tools" in kwargs:
        data["tools"] = kwargs["tools"]
    ...
```

#### 4. tool_calls 格式转换（双向）

**API → LangChain：**

```python
def _convert_tool_calls(self, raw_tool_calls):
    result = []
    for tc in raw_tool_calls:
        result.append({
            "id": tc["id"],
            "name": tc["function"]["name"],
            "args": json.loads(tc["function"]["arguments"])
        })
    return result
```

**LangChain → API：**

```python
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
```

#### 5. 消息转换要保留 tool_calls

```python
elif isinstance(message, AIMessage):
    msg_dict = {"role": "assistant", "content": message.content or ""}
    if message.tool_calls:
        msg_dict["tool_calls"] = self._convert_tool_calls_for_api(message.tool_calls)

elif isinstance(message, ToolMessage):
    msg_dict = {
        "role": "tool",
        "content": message.content,
        "tool_call_id": message.tool_call_id
    }
```

### 为什么不能只用 `_call`？

`_call()` 只能返回 `str`，无法携带 `tool_calls` 字段。要返回完整的 `AIMessage`（包括 `tool_calls`），必须重写 `_generate()`，返回 `ChatResult`。

```python
from langchain_core.outputs import ChatGeneration, ChatResult

def _generate(self, messages, stop=None, run_manager=None, **kwargs):
    ...
    ai_message = AIMessage(
        content=res.get("content", ""),
        tool_calls=tool_calls
    )
    return ChatResult(generations=[ChatGeneration(message=ai_message)])
```

---

## 单元 5：LCEL 管道操作符 `|`

### 核心目标

理解 LangChain Expression Language：用 `|` 把多个 Runnable 组件串联起来。

### 基本用法

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "你是一个{domain}专家"),
    ("human", "请解释{topic}")
])

llm = MyChatModel()
parser = StrOutputParser()

chain = prompt | llm | parser

result = chain.invoke({"domain": "编程", "topic": "Python"})
```

### 数据流转

```
{"domain": "...", "topic": "..."}
    ↓
prompt.invoke() → List[BaseMessage]
    ↓
llm.invoke() → AIMessage
    ↓
parser.invoke() → str
    ↓
最终结果
```

### 每个组件都是 Runnable

| 组件 | 输入 | 输出 |
|------|------|------|
| `ChatPromptTemplate` | `dict` | `List[BaseMessage]` |
| `ChatModel` | `List[BaseMessage]` | `AIMessage` |
| `StrOutputParser` | `AIMessage` / `str` | `str` |
| 自定义 Parser | 任意 | 任意 |

### 自定义 OutputParser

```python
from langchain_core.output_parsers import BaseOutputParser

class UpperCaseParser(BaseOutputParser[str]):
    def parse(self, text: str) -> str:
        return text.upper()

chain = prompt | llm | UpperCaseParser()
```

### 重要理解

- `|` 类似 Unix 管道，前一个的输出作为后一个的输入。
- 组件之间必须类型匹配，否则会报错。
- LangChain 内置组件之间会自动做一些兼容转换（如 `AIMessage` → `str`）。
- 自定义组件时，必须自己保证输入输出格式正确。

---

## 常见错误与解决

| 错误 | 原因 | 解决 |
|------|------|------|
| `NoneType has no attribute 'rstrip'` | `.env` 没读到，变量是 `None` | `load_dotenv()` 指定路径；类初始化时校验 |
| `iter_lines()` 打印生成器对象 | 没遍历生成器 | 用 `for line in response.iter_lines()` |
| `unknown variant 'human'` | 直接用 `message.type` 当 role | HumanMessage 应映射为 `user` |
| `unexpected keyword argument 'stop'` | `_call` 签名不完整 | 添加 `stop`, `run_manager`, `**kwargs` |
| `Can't instantiate abstract class` | 没实现 `_call` | 保留 `_call` 方法 |
| `KeyError: 'choices'` | API 返回错误而非正常响应 | 检查消息格式，尤其是 tool_calls 转换 |
| `unknown variant 'tool_call'` | LangChain 的 tool_calls 格式直接发给 API | 转换 `type: "tool_call"` → `"function"` |
| `ValidationError: should be BaseMessageChunk` | 流式用了 `AIMessage` 而非 `AIMessageChunk` | 流式用 `AIMessageChunk` |

---

## 核心设计思想

1. **Runnable 协议**：所有组件都实现 `invoke` / `stream` / `batch`，可以任意组合。
2. **模板方法模式**：框架定义流程（`invoke` → `_call` / `_generate`），你填写具体实现。
3. **消息抽象**：`BaseMessage` 统一了不同来源的消息，但需要注意 LangChain 格式与 API 格式的转换。
4. **工具调用**：模型只负责"决定"调什么工具，工具的实际执行由外部代码完成，结果再返回给模型总结。
5. **LCEL 组合**：用 `|` 将 prompt、模型、parser 组合成链，使代码更清晰、可复用。

---

## 单元 6：LangGraph 状态图

### 核心目标

把单元 4 手写的 tool calling 循环改造成可自动循环、可记忆的 LangGraph 状态图。

### 关键组件

| 组件 | 作用 |
|------|------|
| `StateGraph` | 定义 Agent 工作流图 |
| `AgentState` | 图的共享状态，通常包含 `messages` |
| `add_messages` | 自动合并各节点返回的消息 |
| `add_node` / `add_edge` | 添加节点和固定流向 |
| `add_conditional_edges` | 根据条件决定下一步 |
| `MemorySaver` | 在同一线程内保存对话历史 |
| `thread_id` | 区分不同对话会话 |

### 图的结构

```
START → call_model → should_continue
                          ↓
              有 tool_calls    没有
                   ↓              ↓
                tools          END
                   ↓
              call_model
                   ↓
                 END
```

### 多工具管理

用字典把工具名映射到工具函数，避免写一堆 `if/else`：

```python
tools_by_name = {
    "get_current_weather": get_current_weather,
    "get_time": get_time,
}
```

### 记忆的关键

```python
from langgraph.checkpoint.memory import MemorySaver

graph = graph_builder.compile(checkpointer=MemorySaver())
config = {"configurable": {"thread_id": "session_1"}}
graph.invoke({"messages": [...]}, config=config)
```

同一个 `thread_id` 的多次调用会自动继承历史状态；不同 `thread_id` 之间互相隔离。

### 常见错误

| 错误 | 原因 | 修复 |
|------|------|------|
| `Checkpointer requires thread_id` | 用了 `MemorySaver` 但 `invoke` 没传 `config` | 每次调用都传 `config={"configurable": {"thread_id": "xxx"}}` |
| 模型不记得之前说过的话 | `thread_id` 每次都不一样 | 保持同一个 `thread_id` |
| 路由返回的节点名不存在 | `should_continue` 返回值和 `add_node` 名不匹配 | 确保返回 `"tools"`、`"__end__"` 等已注册节点名 |

---

## 后续学习方向

1. **LangGraph**：用状态图构建多步 Agent 工作流。
2. **RAG**：文档加载、分块、Embedding、向量检索。
3. **Memory 持久化**：长期记忆、会话管理。
4. **LangSmith**：追踪、调试、评估链和 Agent。

---

## 🎯 下一步学习计划（给后续 AI 参考）

用户的核心目标是**构建个人 AI 助手**。建议按以下顺序推进，每个阶段都产出可运行的代码。

### 阶段 1：LangGraph 入门

- **目标**：把单元 4 的手写 tool calling 循环改造成 LangGraph 状态图。
- **产出**：一个能自动循环调用工具的 Agent，直到不需要工具为止。
- **关键概念**：`StateGraph`、节点、条件边、`checkpointer`。
- **建议文件**：`study/langgraph_agent.py`

### 阶段 2：RAG 知识库

- **目标**：让用户能上传自己的文档（PDF、Markdown、TXT），助手能基于文档回答问题。
- **产出**：支持加载本地文件、切片、生成 Embedding、存入向量库、检索回答。
- **关键组件**：`TextLoader`、`RecursiveCharacterTextSplitter`、`OpenAIEmbeddings` / 本地 Embedding、Chroma / FAISS。
- **建议文件**：`study/rag_knowledge_base.py`

### 阶段 3：长期记忆系统

- **目标**：把 `test_memory.py` 的自动反思记忆升级成更稳定的结构。
- **产出**：
  - 事实记忆：用户偏好、习惯、重要信息。
  - 对话摘要：长对话的压缩摘要。
  - 按主题/时间检索，而不是全部塞进 system prompt。
- **关键概念**：Memory 分类、摘要生成、向量检索记忆。

### 阶段 4：个人助手产品化

- **目标**：把以上能力整合成一个可日常使用的应用。
- **可选形式**：
  - **Gradio 网页**：最轻量，浏览器打开即用。
  - **Streamlit 网页**：更灵活，适合扩展。
  - **本地 API**：FastAPI 提供接口，供其他工具调用。
- **建议文件**：`study/app.py` 或 `api/main.py`

### 阶段 5：高级能力

- 多 Agent 协作（规划 Agent + 执行 Agent + 检查 Agent）。
- 工具错误处理与重试。
- LangSmith 监控与评估。
- 模型输出结构化（JSON mode、Pydantic parser）。

---

## 💬 用户提问时的常见模式

- "帮我优化一下" → 用户希望改进现有代码，但不想大改架构。
- "怎么做" / "如何做" → 用户需要分步骤的指导，适合给出 3-5 步方案。
- "这是什么意思" → 用户遇到新概念，需要用类比 + 代码示例解释。
- "报错" → 用户需要调试帮助，优先让其贴完整 Traceback 和当前代码。
- "我想的是..." → 用户在描述需求，帮助其澄清并转化为技术方案。

---

## ⚠️ 给 AI 的注意事项

- 用户当前项目依赖 `langchain_core`，不要使用已废弃的 `langchain.chains`、`initialize_agent`、`ConversationBufferMemory` 等旧 API。
- 用户习惯把练习文件放在 `study/` 目录，命名规律为 `union_X.py` 或 `test_xxx.py`。
- 用户的环境变量配置在 `d:\projects\XlingAI\.env`，读取时建议用脚本路径定位，不要依赖当前工作目录。
- 解释错误时，先指出最可能的根因，再给出修复代码，最后解释为什么这样修。

5. **多 Agent 系统**：多个模型协作完成复杂任务。
