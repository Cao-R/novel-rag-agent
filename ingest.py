import os
import re
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings  # 推荐使用本地模型
from langchain_core.documents import Document

# === 配置项 ===
CLEAN_NOVELS_DIR = "./clean_novels"  # 存放清洗后文件的目录
DB_PERSIST_DIR = "./local_novel_db"  # 数据库存储路径
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2" # 轻量级模型，支持中文推荐用 'BAAI/bge-m3' 或 'moka-ai/m3e-base'
MODEL_NAME = "./models/BAAI/bge-small-zh-v1___5"  # 本地模型路径




def split_text_by_chapters(text, book_name):
    """
    逻辑：先按“第xxx章”把整本书切开，生成带章节名的 Document 对象列表
    """
    # 匹配 "第xxx章 标题" 的正则
    # 解释：^ 匹配行首，\s* 允许空格，第...章，.* 匹配标题
    chapter_pattern = r"(第[0-9一二三四五六七八九十百千]+[章卷][^\n]*)"
    
    parts = re.split(chapter_pattern, text)
    
    documents = []
    current_chapter = "序章/前言" # 默认第一部分
    
    # re.split 切分后，列表结构是: [前言内容, 标题1, 内容1, 标题2, 内容2...]
    # 所以我们需要跳过偶数项（标题），把它们作为元数据
    
    # 处理第一段（可能是前言）
    if parts[0].strip():
        documents.append(Document(
            page_content=parts[0],
            metadata={"source": book_name, "chapter": current_chapter}
        ))

    # 从索引1开始遍历，步长为2
    for i in range(1, len(parts), 2):
        chapter_title = parts[i].strip()   # 标题 (如 "第一章 陨落的天才")
        chapter_content = parts[i+1].strip() # 内容
        
        if not chapter_content:
            continue
            
        documents.append(Document(
            page_content=chapter_content,
            metadata={
                "source": book_name, 
                "chapter": chapter_title
            }
        ))
    
    return documents

def ingest_to_db():
    # 获取绝对路径，防止相对路径有时候报错
    # 1. 初始化 Embedding 模型 (本地运行，不需要API Key)
    if os.path.exists(MODEL_NAME):
        model_path = os.path.abspath(MODEL_NAME)
    else:
        model_path = MODEL_NAME  # 作为 HuggingFace 模型名使用
    print(f"正在加载本地模型: {model_path}")
    # 如果你在国内下载慢，可以先手动下载模型，然后填本地路径
    embeddings = HuggingFaceEmbeddings(
        model_name=model_path,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. 遍历文件处理
    all_splits = []
    
    if not os.path.exists(CLEAN_NOVELS_DIR):
        print(f"错误：找不到目录 {CLEAN_NOVELS_DIR}")
        return

    for filename in os.listdir(CLEAN_NOVELS_DIR):
        if not filename.endswith(".txt"):
            continue
            
        book_name = filename.replace(".txt", "")
        file_path = os.path.join(CLEAN_NOVELS_DIR, filename)
        
        print(f"正在处理书籍: {book_name}...")
        
        # 读取清洗后的文本
        with open(file_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
            
        # A. 第一层切分：按章节 (保留语义结构)
        chapter_docs = split_text_by_chapters(full_text, book_name)
        print(f"  - 识别到 {len(chapter_docs)} 个章节")
        
        # B. 第二层切分：按长度 (适应模型窗口)
        # chunk_size=500: 每个块约500字符
        # chunk_overlap=100: 重叠100字，防止句子被截断
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=100,
            separators=["\n\n", "\n", "。", "，"]
        )
        
        # 对每个章节进一步切分
        splits = text_splitter.split_documents(chapter_docs)
        
        # C. (可选) 增强内容：把章节名加到正文里，提高检索准确率
        for split in splits:
            # 修改正文，加上 "[斗破苍穹-第一章] " 前缀
            # 这样检索 "斗破第一章" 时更容易命中
            header = f"[{split.metadata['source']} - {split.metadata['chapter']}]\n"
            split.page_content = header + split.page_content
            
        all_splits.extend(splits)
        print(f"  - 最终切分为 {len(splits)} 个向量块")

    # 3. 存入 ChromaDB
    if all_splits:
        print(f"正在将 {len(all_splits)} 个数据块写入数据库 (./local_novel_db)...")
        # 这一步会创建/更新本地数据库文件夹
        vectorstore = Chroma.from_documents(
            documents=all_splits,
            embedding=embeddings,
            persist_directory=DB_PERSIST_DIR,
            collection_name=book_name
        )
        print("入库成功！数据库已保存。")
    else:
        print("没有数据需要入库。")

if __name__ == "__main__":
    ingest_to_db()