import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# 1. 指定本地模型路径 (跟入库时一样)
model_path = os.path.abspath("./models/BAAI/bge-small-zh-v1___5")
embedding_function = HuggingFaceEmbeddings(
    model_name=model_path,
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# 2. 加载已经存在的数据库
db_path = "./local_novel_db"
if not os.path.exists(db_path):
    print("错误：数据库路径不存在！请先运行 ingest.py")
    exit()

vectorstore = Chroma(
    persist_directory=db_path, 
    embedding_function=embedding_function
)

# 3. 测试搜索
query = "丁寿的绝招是什么？"  # 换成你小说里相关的问题
print(f"正在查询: {query}")

# k=3 表示找最相似的3个片段
results = vectorstore.similarity_search(query, k=3)

print(f"\n找到 {len(results)} 条相关结果：\n")
for i, doc in enumerate(results):
    print(f"--- 结果 {i+1} ---")
    print(f"来源: {doc.metadata.get('source')} | 章节: {doc.metadata.get('chapter')}")
    print(f"内容片段: {doc.page_content[:100]}...") # 只打印前100字预览
    print("-" * 30)