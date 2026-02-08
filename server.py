import os
import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, List, Optional, AsyncGenerator, Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# LangChain / LangGraph 相关
from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

# 导入 Tavily (尽量使用新版导入方式，避免警告)
try:
    from langchain_tavily import TavilySearchResults
except ImportError:
    from langchain_community.tools.tavily_search import TavilySearchResults

from langchain_core.prompts import ChatPromptTemplate

from graph_utils import graph_search_with_llm

# 配置 API Key
os.environ["TAVILY_API_KEY"] = "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"

# --- 1. 定义资源加载器 (Lifespan 模式) ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    # 【启动阶段】：加载模型和数据库
    print("正在初始化本地资源 (Lifespan)...")
    
    # 初始化 Embeddings
    # 请确保路径正确，如果不正确请修改 path
    model_path = os.path.abspath("./models/BAAI/bge-small-zh-v1___5")
    
    # 简单的路径检查
    if not os.path.exists(model_path):
        # 尝试备用路径（有时候文件夹名字不一样）
        alt_path = os.path.abspath("./models/bge-small-zh-v1.5")
        if os.path.exists(alt_path):
            model_path = alt_path
    
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'}
    )
    
    # 加载向量库
    vectorstore = Chroma(
        persist_directory="./local_novel_db", 
        embedding_function=embeddings
    )
    
    # 创建 Agent 运行对象
    agent_app = create_agent_graph(vectorstore)
    
    app.state.agent_app = agent_app  # 直接赋值给 app.state
    app.state.vectorstore = vectorstore
    
    # 将资源存入 app.state，方便在所有路由中访问
    yield {
        "vectorstore": vectorstore,
        "agent_app": agent_app
    }
    
    # 【关闭阶段】：清理资源
    print("正在关闭服务，释放资源...")

# --- 2. 状态定义 ---
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# --- 3. Pydantic V2 结构化输出定义 ---
class GradeDocuments(BaseModel):
    """逻辑评分结果"""
    binary_score: str = Field(description="相关性评分: 'yes' 或 'no'")

class QueryRewrite(BaseModel):
    """查询重写结果"""
    rewritten_query: str = Field(description="为互联网搜索优化的改写问题")

# --- 4. 构建 V2 智商增强型图 ---
def create_agent_graph(vectorstore: Chroma):
    # 初始化 Ollama 模型
    llm = ChatOllama(model="qwen2.5:7b", format="json", temperature=0, timeout=120.0)
    
    # 联网工具
    web_search_tool = TavilySearchResults(k=3)

    # --- 节点 A: 检索 ---
    async def retrieve_node(state: AgentState):
        print("--- [Node: Retrieve] 正在从本地向量库检索... ---")
        question = state["messages"][0].content
        # 异步执行检索
        loop = asyncio.get_event_loop()
        graph_text = await loop.run_in_executor(
            None, lambda: graph_search_with_llm(question, llm)
        )
        docs = await loop.run_in_executor(
            None, lambda: vectorstore.similarity_search(question, k=3)
        )

        context_parts = []
        if graph_text:
            context_parts.append("[Graph]\n" + graph_text)
        if docs:
            vector_context = "\n\n".join([f"内容: {d.page_content}" for d in docs])
            context_parts.append("[Vector]\n" + vector_context)
        context = "\n\n".join(context_parts)
        # 将检索到的资料临时存入 SystemMessage 传给下一个节点
        return {"messages": [SystemMessage(content=context, name="context_data")]}

    # --- 节点 B: 评分与决策 (CRAG 核心) ---
    async def grade_node(state: AgentState) -> Literal["generate", "web_search"]:
        print("--- [Node: Grade] 正在评估文档相关性... ---")
        question = state["messages"][0].content
        context = state["messages"][-1].content # 获取上一个节点的检索内容

        # 使用 Pydantic V2 进行强类型约束输出
        structured_llm = llm.with_structured_output(GradeDocuments)
        
        # 去掉 f-string，使用 LangChain 标准占位符
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个严格的质检员。判断提供的资料是否能回答用户问题。只返回 JSON。"),
            ("human", "用户问题: {question} \n\n 检索到的资料: {context}")  
        ])
        
        chain = prompt | structured_llm
        try:
            # 传入变量字典
            score = await chain.ainvoke({"question": question, "context": context})
            if score.binary_score == "yes":
                print("--- [Result] 资料相关，准备回答 ---")
                return "generate"
            else:
                print("--- [Result] 资料不相关，触发联网搜索 ---")
                return "web_search"
        except Exception as e:
            print(f"评分出错，默认跳转生成: {e}")
            return "generate"

    # --- 节点 C: 联网搜索 ---
    async def web_search_node(state: AgentState):
        print("--- [Node: Web Search] 正在执行联网补位... ---")
        question = state["messages"][0].content
        
        # 进阶：先改写问题再搜
        rewrite_llm = llm.with_structured_output(QueryRewrite)
        
        # 【修复 2】：去掉 f-string
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", "将用户的口语化问题改写为适合搜索引擎的关键词。只返回 JSON。"),
            ("human", "原始问题: {question}")
        ])
        
        # 【修复 3】：必须传入 {"question": question}，之前传的是空 {}
        rewrite_res = await (rewrite_prompt | rewrite_llm).ainvoke({"question": question})
        
        print(f"   (改写后查询: {rewrite_res.rewritten_query})")
        search_results = await web_search_tool.ainvoke(rewrite_res.rewritten_query)
        
        return {"messages": [SystemMessage(content=str(search_results), name="web_data")]}

    # --- 节点 D: 最终回答 ---
    async def generate_node(state: AgentState):
        print("--- [Node: Generate] 正在汇总回答... ---")
        question = state["messages"][0].content
        # 寻找之前节点存入的消息
        context = ""
        for msg in reversed(state["messages"]):
            if isinstance(msg, SystemMessage) and msg.name in ["context_data", "web_data"]:
                context = msg.content
                break

        # 【确认】：这里没有用 f-string，是正确的
        gen_prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个精通小说的大模型助手。请根据提供的资料回答问题。如果资料中没有相关信息，请诚实说明。若资料中包含 Cypher 或 Path，请在回答末尾保留。"),
            ("human", "资料内容如下:\n{context}\n\n用户问题: {question}")
        ])
        
        response = await (gen_prompt | llm).ainvoke({"context": context, "question": question})
        return {"messages": [response]}

    # --- 3. 组装图 (Graph Construction) ---
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve_node", retrieve_node)
    workflow.add_node("web_search_node", web_search_node)
    workflow.add_node("generate_node", generate_node)

    workflow.add_edge(START, "retrieve_node")
    
    # 动态路由
    workflow.add_conditional_edges(
        "retrieve_node",
        grade_node,
        {
            "generate": "generate_node",
            "web_search": "web_search_node"
        }
    )
    
    workflow.add_edge("web_search_node", "generate_node")
    workflow.add_edge("generate_node", END)

    return workflow.compile()

# --- 5. 定义 FastAPI App ---
app = FastAPI(title="Novel RAG Agent", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 6. 数据模型 ---
class ChatRequest(BaseModel):
    query: str
    model_config = {
        "json_schema_extra": {
            "example": {
                "query": "丁寿的绝招是什么？"
            }
        }
    }

# --- 7. 路由接口 ---

@app.post("/chat")
async def chat_endpoint(request: ChatRequest, http_request: Request):
    agent_app = http_request.state.agent_app
    inputs = {"messages": [HumanMessage(content=request.query)]}
    result = await agent_app.ainvoke(inputs)
    return {"answer": result["messages"][-1].content}

@app.get("/chat/stream")
async def chat_stream_endpoint(query: str, http_request: Request):
    agent_app = http_request.state.agent_app
    
    async def event_generator():
        inputs = {"messages": [HumanMessage(content=query)]}
        async for event in agent_app.astream(inputs, stream_mode="values"):
            if "messages" in event: 
                msg = event["messages"][-1]
                if msg.type == "ai" and msg.content:
                    yield f"data: {msg.content}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)