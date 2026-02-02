# Novel RAG Agent - 垂直领域小说智能体

基于 **LangGraph** 与 **Qwen2.5** 构建的异步多智能体 RAG 系统。针对长篇小说情节复杂、实体关系繁多的痛点，实现了具备**自我修正 (CRAG)** 与 **联网补位** 能力的高性能问答助手。

## 核心特性

- **🕷️ 高并发异步架构**: 基于 **FastAPI** + **Asyncio** 构建，支持 **SSE 流式输出**，针对向量库检索进行了 I/O 优化。
- **🧠 CRAG 高级推理**: 基于 **LangGraph** 实现 "检索-评估-补位" 状态机。具备自我反思能力，当本地知识不足时自动触发 **Tavily** 联网搜索。
- **🔍 混合检索**: 集成 **ChromaDB** 与 **BGE-M3** 模型，支持小说章节级切分与元数据过滤。
- **🛡️ 稳健工程**: 遵循 **Pydantic V2** 规范，严格的数据校验与结构化输出 (Structured Output)。
- **🎭 角色扮演 (SFT)**: (可选) 支持加载经 LoRA 微调的 Qwen2.5 模型，提供沉浸式对话体验。

## 技术栈

- **LLM**: Ollama (Qwen2.5-7B), LangChain
- **Backend**: Python 3.12, FastAPI, Uvicorn
- **Agent**: LangGraph, CRAG Architecture
- **Database**: ChromaDB (Vector Store)
- **Tools**: Tavily Web Search
- **Frontend**: Streamlit

## 项目结构

```bash
├── server.py           # FastAPI 后端核心 (Agent逻辑/路由)
├── client.py           # Streamlit 前端界面
├── ingest.py           # 数据处理与向量入库脚本
├── models/             # 本地 Embedding 模型 
├── local_novel_db/     # 向量数据库文件 (不上传)
└── requirements.txt    # 项目依赖
