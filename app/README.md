# XlingAI 产品化界面

基于 `study/union_8.py` 的成果，用 Gradio 提供一个可交互的网页界面。

## 目录结构

```
app/
├── __init__.py
├── agent.py       # Agent 核心逻辑（长期记忆 + 工具 + LangGraph）
├── main.py        # Gradio 界面入口
└── README.md
```

## 运行

在项目根目录执行：

```bash
python -m app.main
```

或直接：

```bash
python app/main.py
```

启动后打开浏览器访问 `http://127.0.0.1:7860`。

## 功能

- 左侧对话窗口：与助手多轮对话
- 右侧长期记忆面板：查看当前保存的用户偏好和事实
- 短期记忆：同一浏览器会话内的上下文由 LangGraph MemorySaver（`thread_id="web_user"`）自动维护
- 长期记忆：反思节点在每轮对话后从对话里提取事实，保存到 `study/memory.json`

## 依赖

依赖已经在项目根目录的 `requirements.txt` 里：

- gradio
- langgraph
- langchain-core
- langchain-huggingface
- langchain-chroma
- langchain-text-splitters
- python-dotenv
- requests
