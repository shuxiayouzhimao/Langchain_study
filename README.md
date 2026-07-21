# XlingAI

个人 AI 助手学习项目，基于 LangChain Core 和 DeepSeek API 构建。

## 项目目标

构建一个能在日常生活中提供帮助的私人 AI 助手，具备：

- 长期记忆与自动学习能力
- 本地知识库（RAG）
- 工具调用能力
- 可本地部署的交互界面

## 目录结构

```
XlingAI/
├── study/                      # 学习练习代码
│   ├── union_1.py              # 纯 requests 调用 DeepSeek API（流式）
│   ├── union_2.py              # LangChain 消息对象体系
│   ├── union_3.py              # 手写 MyChatModel（继承 SimpleChatModel）
│   ├── union_4.py              # Tool Calling 完整实现
│   ├── union_5.py              # LCEL 管道与自定义 Parser
│   └── langchain_study_notes.md # 学习笔记（AI 可读）
├── .env                        # 环境变量（不提交到 Git）
├── .env.example                # 环境变量模板
├── .gitignore
├── requirements.txt
└── README.md
```

## 环境配置

1. 复制环境变量模板并填写：

```bash
cp .env.example .env
```

2. 在 `.env` 中配置 DeepSeek API：

```env
DEEPSEEK_API_KEY=your_api_key_here
DEEPSEEK_API_OPENAI_URL=https://api.deepseek.com
DEEPSEEK_API_MODEL=deepseek-chat
```

3. 安装依赖：

```bash
pip install -r requirements.txt
```

## 运行示例

```bash
# 运行纯 API 调用示例
python study/union_1.py

# 运行消息对象示例
python study/union_2.py

# 运行自定义 ChatModel 示例
python study/union_3.py

# 运行工具调用示例
python study/union_4.py

# 运行 LCEL 管道示例
python study/union_5.py
```

## 学习路线

详见 [`study/langchain_study_notes.md`](study/langchain_study_notes.md)。

## 注意事项

- `.env` 文件包含 API 密钥，请勿提交到 GitHub。
- `venv/` 目录为本地虚拟环境，已加入 `.gitignore`。

## License

MIT
