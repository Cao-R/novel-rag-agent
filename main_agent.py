import os
from typing import Annotated, Literal, TypedDict
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama

from graph_utils import graph_search_with_llm

# --- 1. 准备工作：加载数据库 ---
print("正在加载数据库...")
model_path = os.path.abspath("./models/BAAI/bge-small-zh-v1___5")
embedding_func = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

vectorstore = Chroma(
    persist_directory="./local_novel_db", 
    embedding_function=embedding_func
)

# --- 2. 初始化 LLM ---
# 温度设为 0，让回答更基于事实
llm = ChatOllama(model="qwen2.5:7b", temperature=0, timeout=120.0)

# --- 3. 定义工具 (Tool) ---
# 这是 Agent 与数据库交互的接口

@tool
def search_graph(query: str):
    """
    用于图谱查询（人物关系、事件、章节等）。
    """
    result = graph_search_with_llm(query, llm)
    return result if result else "图谱未命中。"

@tool
def search_novel(query: str, book_name: str = None):
    """
    用于查询小说内容。
    参数:
    - query: 具体的问题或关键词。
    - book_name: (可选) 书名。如果用户明确指出了是哪本书，必须传入此参数。
                 例如："斗破苍穹", "凡人修仙传"。
    """
    filter_rule = None
    if book_name:
        # 这里的 "source" 对应我们入库时的 metadata 字段
        filter_rule = {"source": book_name}
        print(f"   [工具调用] 正在书籍《{book_name}》中搜索: {query}")
    else:
        print(f"   [工具调用] 正在所有书籍中搜索: {query}")

    # 执行搜索，返回最相关的 5 个片段
    results = vectorstore.similarity_search(query, k=5, filter=filter_rule)
    
    # 拼接结果给大模型看
    context = ""
    for doc in results:
        source = doc.metadata.get('source', '未知来源')
        chapter = doc.metadata.get('chapter', '未知章节')
        context += f"《{source}》[{chapter}]:\n{doc.page_content}\n\n"
    
    return context if context else "未找到相关内容。"

# --- 4. 构建 Agent 流程 (LangGraph) ---   

# 定义状态
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]

# 绑定工具
tools = [search_graph, search_novel]
llm_with_tools = llm.bind_tools(tools)

# 节点逻辑：调用模型
def agent_node(state: AgentState):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

# 逻辑判断：继续调用工具还是结束
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 组装图
workflow = StateGraph(AgentState)
workflow.add_node("agent", agent_node)
workflow.add_node("tools", ToolNode(tools))

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent") # 工具用完后回传给 Agent 生成答案

app = workflow.compile()

# --- 4. 交互式运行 ---
if __name__ == "__main__":
    print("\n=== 小说多智能体助手已启动 (输入 'q' 退出) ===")
    
    while True:
        user_input = input("\n请输入问题: ")
        if user_input.lower() in ['q', 'quit', 'exit']:
            break
            
        initial_state = {
            "messages": [
                SystemMessage(content=(
                    "你是一个精通小说的专家。优先使用图谱回答人物关系、事件、章节等结构化问题。"
                    "图谱无结果时再使用向量检索补充。"
                    "如果图谱工具返回了 Cypher 或 Path，请在回答末尾原样保留。"
                )),
                HumanMessage(content=user_input)
            ]
        }
        
        # 流式输出，可以看到 Agent 的思考过程
        try:
            final_response = app.invoke(initial_state)
            print("\n>>> 回答:")
            print(final_response["messages"][-1].content)
        except Exception as e:
            print(f"发生错误: {e}")