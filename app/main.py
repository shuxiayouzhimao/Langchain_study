"""XlingAI 个人助手 Gradio 界面。

运行方式（在项目根目录 d:/projects/XlingAI 执行）：

    python -m app.main

或直接执行文件：

    python app/main.py

启动后浏览器打开 http://127.0.0.1:7860 即可对话。
"""

import sys
import io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

import os
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import gradio as gr
from langchain_core.messages import HumanMessage

from app.agent import graph, memory_store


def chat(message: str, history: list) -> str:
    """处理单轮用户输入，返回助手回复。

    history 由 Gradio 自动维护，这里不使用——短期记忆已由
    LangGraph 的 MemorySaver（thread_id="web_user"）负责。
    """
    config = {"configurable": {"thread_id": "web_user"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    return result["messages"][-1].content


def get_memory_display() -> str:
    """把长期记忆格式化成 Markdown 显示。"""
    facts = memory_store.get_facts()
    if not facts:
        return "*（暂无长期记忆）*"
    lines = [f"- **[{f['category']}]** {f['content']}" for f in facts]
    return "\n".join(lines)


with gr.Blocks(title="XlingAI 个人助手") as demo:
    gr.Markdown("# XlingAI 个人助手\n具备长期记忆、知识库、工具调用能力的 AI 助手。")

    with gr.Row():
        with gr.Column(scale=3):
            chat_interface = gr.ChatInterface(
                fn=chat,
                type="messages",
                examples=[
                    "我叫小明，我喜欢吃火锅",
                    "我叫什么？喜欢吃什么？",
                    "厦门天气怎么样？",
                    "员工年假有多少天？",
                ],
            )

        with gr.Column(scale=1):
            gr.Markdown("## 长期记忆")
            memory_display = gr.Markdown(get_memory_display())
            refresh_btn = gr.Button("刷新记忆")
            refresh_btn.click(fn=get_memory_display, outputs=memory_display)


if __name__ == "__main__":
    demo.launch()
