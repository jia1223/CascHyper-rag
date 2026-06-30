"""
Simple RAG System - 基于 HyperRAG v7.3 简化的普通 RAG 实现

架构：
- 文档加载 → Token级切块 → Embedding向量化 → 向量检索 → LLM生成答案

特点：
- 简单直接的向量相似度检索
- 无复杂的主题发现、级联更新、多跳检索
- 适合快速部署和基础问答场景
"""

import asyncio
import json
import re
import numpy as np
import os
import sys
import pickle
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# ============================================================================
# 0. Global Configuration & API
# ============================================================================

sys.path.append("./Hyper-RAG")
from hyperrag.llm import siliconflow_complete, siliconcloud_embedding
from hyperrag.operate import chunking_by_token_size

os.environ["OPENAI_API_KEY"] = os.getenv("SILICONFLOW_API_KEY", "")

def normalize(v: np.ndarray) -> np.ndarray:
    """L2 Normalization"""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v

# --- Embedding API ---
async def get_embedding_api(texts: List[str]) -> List[np.ndarray]:
    """调用 Embedding API (BAAI/bge-m3)"""
    if not texts: 
        return []
    api_key = os.environ.get("OPENAI_API_KEY")
    embeddings = await siliconcloud_embedding(
        texts=texts, 
        model="BAAI/bge-m3", 
        base_url="https://api.siliconflow.cn/v1/embeddings", 
        api_key=api_key
    )
    return [normalize(np.array(e)) for e in embeddings]

# --- Completion API ---
async def get_completion_api(prompt: str) -> str:
    """调用 LLM API (DeepSeek-V3)"""
    return await siliconflow_complete(prompt)

# ============================================================================
# 1. Data Structures
# ============================================================================

@dataclass
class ChunkNode:
    """文档块节点"""
    chunk_id: int
    text: str
    embedding: Optional[np.ndarray] = None

# ============================================================================
# 2. Document Processing
# ============================================================================

def load_markdown_file(file_path: str) -> str:
    """加载 Markdown 文件内容，移除 YAML front matter"""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    if content.startswith('---'):
        end_idx = content.find('---', 3)
        if end_idx != -1:
            content = content[end_idx + 3:].strip()
    return content

def clean_markdown_text(text: str) -> str:
    """清理 Markdown 文本"""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)  # 移除图片
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)  # 保留链接文字
    text = re.sub(r'<[^>]+>', '', text)  # 移除 HTML 标签
    text = re.sub(r'\n{3,}', '\n\n', text)  # 压缩多余空行
    return text.strip()

def chunk_document(text: str, max_token_size: int = 400, overlap_token_size: int = 50) -> List[Dict]:
    """将长文档切分为多个块 (Token级切分)"""
    chunks = chunking_by_token_size(
        content=text,
        max_token_size=max_token_size,
        overlap_token_size=overlap_token_size
    )
    result = []
    for idx, chunk in enumerate(chunks):
        if isinstance(chunk, dict):
            result.append({"content": chunk.get("content", chunk.get("text", "")), "chunk_index": idx})
        else:
            result.append({"content": str(chunk), "chunk_index": idx})
    return result

# ============================================================================
# 3. Cache Utilities
# ============================================================================

CACHE_DIR = "./simple_rag_cache"

def get_cache_path(content: str, prefix: str = "simple_rag") -> str:
    """根据内容生成缓存文件路径"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
    return os.path.join(CACHE_DIR, f"{prefix}_{content_hash}.pkl")

def save_index(chunks: Dict[int, ChunkNode], cache_path: str):
    """保存索引到缓存文件"""
    with open(cache_path, 'wb') as f:
        pickle.dump(chunks, f)
    print(f"[Cache] 已保存索引到 {cache_path}")

def load_index(cache_path: str) -> Optional[Dict[int, ChunkNode]]:
    """从缓存文件加载索引"""
    if os.path.exists(cache_path):
        with open(cache_path, 'rb') as f:
            chunks = pickle.load(f)
        print(f"[Cache] 从 {cache_path} 加载了 {len(chunks)} 个文档块")
        return chunks
    return None

# ============================================================================
# 4. Simple RAG Engine
# ============================================================================

class SimpleRAG:
    """
    普通 RAG 系统
    
    流程：
    1. 索引：文档 → 切块 → Embedding
    2. 检索：Query Embedding → 向量相似度 → Top-K 块
    3. 生成：Context + Query → LLM → Answer
    """
    
    def __init__(self):
        self.chunks: Dict[int, ChunkNode] = {}
    
    # =========================================================================
    # Phase 1: Index Construction
    # =========================================================================
    
    async def build_index(self, raw_text: str, max_token_size: int = 400, overlap_token_size: int = 50):
        """
        构建索引
        
        1. 将文档切分为块
        2. 为每个块生成 Embedding
        """
        print(f"\n{'='*60}")
        print("[Build] Simple RAG Indexing Pipeline")
        print(f"{'='*60}")
        
        # Step 1: 切块
        print(f"\n[Step 1] Chunking document...")
        chunk_data = chunk_document(raw_text, max_token_size=max_token_size, overlap_token_size=overlap_token_size)
        chunk_texts = [c["content"] for c in chunk_data]
        print(f"  -> {len(chunk_texts)} chunks created")
        
        # Step 2: 生成 Embedding
        print(f"\n[Step 2] Generating embeddings...")
        chunk_vecs = await get_embedding_api(chunk_texts)
        
        for i, (txt, vec) in enumerate(zip(chunk_texts, chunk_vecs)):
            self.chunks[i] = ChunkNode(chunk_id=i, text=txt, embedding=vec)
        
        print(f"\n[Index Ready] Total: {len(self.chunks)} chunks indexed")
    
    # =========================================================================
    # Phase 2: Retrieval
    # =========================================================================
    
    async def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        检索相关文档块
        
        1. 生成 Query Embedding
        2. 计算与所有块的余弦相似度
        3. 返回 Top-K 结果
        """
        print(f"\n{'='*60}")
        print(f"[Search] Query: {query}")
        print(f"{'='*60}")
        
        # Step 1: Query Embedding
        print(f"\n[Step 1] Generating query embedding...")
        q_vec = (await get_embedding_api([query]))[0]
        
        # Step 2: 计算相似度
        print(f"\n[Step 2] Computing similarities...")
        results = []
        for cid, chunk in self.chunks.items():
            # 余弦相似度 (向量已归一化，直接点积)
            score = float(np.dot(q_vec, chunk.embedding))
            results.append({
                "chunk_id": cid,
                "text": chunk.text,
                "score": score
            })
        
        # Step 3: 排序并返回 Top-K
        results.sort(key=lambda x: x["score"], reverse=True)
        top_results = results[:top_k]
        
        print(f"\n[Results] Top {len(top_results)} chunks retrieved:")
        for i, res in enumerate(top_results):
            print(f"  {i+1}. [Score: {res['score']:.4f}] {res['text'][:60]}...")
        
        return top_results
    
    # =========================================================================
    # Phase 3: Generation
    # =========================================================================
    
    async def generate_answer(self, query: str, search_results: List[Dict]) -> str:
        """
        基于检索结果生成答案
        
        1. 构建上下文
        2. 调用 LLM 生成答案
        """
        print(f"\n[Generation] Synthesizing answer...")
        
        if not search_results:
            return "抱歉，未找到相关信息。"
        
        # 构建上下文
        context_parts = []
        for i, res in enumerate(search_results):
            context_parts.append(f"[文档片段 {i+1}] (相关度: {res['score']:.2f})\n{res['text']}")
        
        context = "\n\n".join(context_parts)
        
        # 构建 Prompt
        prompt = f"""你是一个专业的问答助手。请根据以下参考文档回答用户的问题。

**重要规则：**
1. 只根据提供的文档内容回答，不要编造信息
2. 如果文档中没有相关信息，请明确说明"根据提供的文档，未找到相关信息"
3. 回答要简洁、准确、有条理
4. 可以引用文档片段编号来标注信息来源

=== 参考文档 ===
{context}

=== 用户问题 ===
{query}

请回答："""
        
        answer = await get_completion_api(prompt)
        return answer
    
    # =========================================================================
    # Convenience Method
    # =========================================================================
    
    async def query(self, question: str, top_k: int = 5) -> str:
        """
        一站式问答接口
        
        检索 + 生成
        """
        results = await self.search(question, top_k=top_k)
        answer = await self.generate_answer(question, results)
        return answer

# ============================================================================
# 5. Main Execution
# ============================================================================

async def main():
    print("=" * 60)
    print("Simple RAG System")
    print("=" * 60)
    
    # 加载文档
    md_file_path = "./honglong.md"
    
    print(f"\n[文档] 加载文件: {md_file_path}")
    raw_text = load_markdown_file(md_file_path)
    cleaned_text = clean_markdown_text(raw_text)
    print(f"[文档] 原始长度: {len(raw_text)} 字符, 清理后: {len(cleaned_text)} 字符")
    
    # 创建 RAG 引擎
    rag = SimpleRAG()
    
    # 检查缓存
    cache_path = get_cache_path(cleaned_text)
    cached_chunks = load_index(cache_path)
    
    if cached_chunks:
        rag.chunks = cached_chunks
        print("[缓存] 使用缓存数据，跳过索引构建")
    else:
        print("\n[缓存] 未找到缓存，开始构建索引...")
        await rag.build_index(
            cleaned_text,
            max_token_size=400,
            overlap_token_size=50
        )
        save_index(rag.chunks, cache_path)
    
    # 测试查询
    print("\n" + "=" * 60)
    print("执行查询测试")
    print("=" * 60)
    
    queries = [
        "总结一下整体内容"
    ]
    
    for q in queries:
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        
        answer = await rag.query(q, top_k=10)
        
        print(f"\n{'='*60}")
        print("Answer:")
        print(f"{'='*60}")
        print(answer)

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())
