"""XlingAI 个人助手 Gradio 界面。

运行方式（在项目根目录 d:/projects/XlingAI 执行）：

    python -m app.main

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


# ========== 业务逻辑 ==========
def chat(message: str, history: list) -> str:
    """处理单轮用户输入，返回助手回复。"""
    config = {"configurable": {"thread_id": "web_user"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )
    return result["messages"][-1].content


def get_memory_display() -> str:
    """把长期记忆格式化成漂亮的 Markdown。"""
    facts = memory_store.get_facts()
    if not facts:
        return (
            "<div class='memory-empty'>"
            "🌱<br><br>还没有任何记忆<br>"
            "<span style='opacity:0.6;font-size:0.9em'>对话结束后我会自动记住重要信息</span>"
            "</div>"
        )

    # 按 category 分组
    by_category = {}
    for f in facts:
        by_category.setdefault(f["category"], []).append(f)

    category_icons = {
        "user_info": "👤",
        "preference": "❤️",
        "fact": "📝",
        "summary": "📚",
        "general": "💡",
    }

    parts = []
    for cat, items in by_category.items():
        icon = category_icons.get(cat, "🔖")
        parts.append(f"### {icon} {cat}")
        for f in items:
            parts.append(f"- {f['content']}")
        parts.append("")

    parts.append(f"---\n<sub>💾 共 {len(facts)} 条记忆</sub>")
    return "\n".join(parts)


def clear_memory() -> str:
    """清空长期记忆。"""
    memory_store.facts = []
    memory_store._save()
    return get_memory_display()


# ========== 界面样式 ==========
CUSTOM_CSS = """
/* 整体背景 */
.gradio-container {
    background: linear-gradient(135deg, #f5f7fa 0%, #e8ecf3 100%) !important;
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', 'PingFang SC',
                 'Hiragino Sans GB', 'Microsoft YaHei', sans-serif !important;
}

.dark .gradio-container {
    background: linear-gradient(135deg, #1a1d29 0%, #2d3142 100%) !important;
}

/* 顶部渐变 header */
#app-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 28px 32px;
    border-radius: 16px;
    margin-bottom: 20px;
    color: white;
    box-shadow: 0 8px 24px rgba(102, 126, 234, 0.25);
}
#app-header h1 {
    margin: 0 !important;
    color: white !important;
    font-size: 1.8em !important;
    font-weight: 700 !important;
}
#app-header p {
    margin: 8px 0 0 0 !important;
    color: rgba(255,255,255,0.9) !important;
    font-size: 0.95em !important;
}

/* 卡片风格 */
.card {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.04);
}
.dark .card {
    background: #2d3142;
    border-color: rgba(255,255,255,0.08);
}

/* 聊天区域 */
.chatbot {
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 2px 12px rgba(0,0,0,0.04) !important;
}

/* 记忆面板 */
#memory-panel {
    background: white;
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    border: 1px solid rgba(0,0,0,0.04);
    min-height: 500px;
}
.dark #memory-panel {
    background: #2d3142;
    border-color: rgba(255,255,255,0.08);
}
#memory-panel h2, #memory-panel h3 {
    margin-top: 0 !important;
}

.memory-empty {
    text-align: center;
    padding: 60px 20px;
    color: rgba(0,0,0,0.5);
    font-size: 1.1em;
}
.dark .memory-empty {
    color: rgba(255,255,255,0.5);
}

/* 按钮 */
button.primary {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    border: none !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: transform 0.15s ease !important;
}
button.primary:hover {
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3) !important;
}

button.secondary {
    background: transparent !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    border-radius: 10px !important;
}

/* 示例按钮 */
.example-btn {
    background: rgba(102, 126, 234, 0.08) !important;
    border: 1px solid rgba(102, 126, 234, 0.2) !important;
    color: #5568d3 !important;
    border-radius: 20px !important;
    padding: 6px 16px !important;
    font-size: 0.9em !important;
}
"""


# ========== 界面 ==========
theme = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="purple",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(
    body_background_fill="*neutral_50",
    block_background_fill="white",
    block_border_width="0px",
    block_shadow="0 2px 12px rgba(0,0,0,0.04)",
    block_radius="16px",
    button_primary_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #7188ee 0%, #8055b0 100%)",
)


with gr.Blocks(title="XlingAI 个人助手", theme=theme, css=CUSTOM_CSS) as demo:

    # 顶部 header
    gr.HTML(
        """
        <div id="app-header">
            <h1>✨ XlingAI 个人助手</h1>
            <p>具备长期记忆 · 本地知识库 · 工具调用能力的 AI 助手</p>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        # 左侧：对话
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                type="messages",
                height=520,
                show_label=False,
                avatar_images=(
                    None,  # 用户默认
                    "https://api.dicebear.com/7.x/bottts/svg?seed=xlingai&backgroundColor=667eea",
                ),
                render_markdown=True,
                elem_classes=["chatbot"],
            )

            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="想聊点什么？（Enter 发送）",
                    show_label=False,
                    scale=8,
                    lines=1,
                    max_lines=4,
                    autofocus=True,
                )
                send_btn = gr.Button("发送", variant="primary", scale=1, min_width=80)

            with gr.Row():
                gr.Examples(
                    examples=[
                        "我叫小明，我喜欢吃火锅",
                        "我叫什么？喜欢吃什么？",
                        "厦门天气怎么样？",
                        "员工年假有多少天？",
                    ],
                    inputs=msg_input,
                    label="💡 试试这些",
                )

        # 右侧：记忆面板
        with gr.Column(scale=1, min_width=280):
            with gr.Group(elem_id="memory-panel"):
                gr.Markdown("## 🧠 长期记忆")
                memory_display = gr.Markdown(get_memory_display())

                with gr.Row():
                    refresh_btn = gr.Button("🔄 刷新", size="sm", variant="secondary")
                    clear_btn = gr.Button("🗑️ 清空", size="sm", variant="secondary")

    # 事件绑定
    def respond(message, history):
        if not message or not message.strip():
            return history, "", get_memory_display()
        history = history or []
        history.append({"role": "user", "content": message})
        reply = chat(message, history)
        history.append({"role": "assistant", "content": reply})
        return history, "", get_memory_display()

    send_btn.click(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input, memory_display],
    )
    msg_input.submit(
        fn=respond,
        inputs=[msg_input, chatbot],
        outputs=[chatbot, msg_input, memory_display],
    )

    refresh_btn.click(fn=get_memory_display, outputs=memory_display)
    clear_btn.click(fn=clear_memory, outputs=memory_display)


if __name__ == "__main__":
    demo.launch(inbrowser=True)
