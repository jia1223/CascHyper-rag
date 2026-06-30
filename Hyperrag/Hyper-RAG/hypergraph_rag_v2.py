#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
新版超图RAG实现脚本
基于用户图片思路重新设计的超图RAG第一步实现

主要功能：
1. 文本切块 - 将输入文本分割成合适大小的块
2. 句子级实体提取 - 对每个句子提取多个实体
3. 实体聚合与超边构建 - 将相关实体聚合形成超边
4. 问题实体提取 - 对输入问题提取实体并构建超边
5. 超边匹配检索 - 使用问题超边匹配文档超边进行检索
"""

import asyncio
import json
import re
import logging
from typing import List, Dict, Set, Tuple, Any
from collections import defaultdict, Counter
from dataclasses import dataclass, field
import jieba
import jieba.posseg as pseg
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Entity:
    """实体类"""
    name: str
    entity_type: str
    description: str = ""
    source_sentence: str = ""
    chunk_id: str = ""
    
@dataclass
class Hyperedge:
    """超边类"""
    entities: Set[str]
    weight: float = 1.0
    description: str = ""
    source_chunk: str = ""
    edge_type: str = "co-occurrence"  # 共现、语义相关等

@dataclass
class TextChunk:
    """文本块类"""
    id: str
    content: str
    sentences: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    hyperedges: List[Hyperedge] = field(default_factory=list)

class HypergraphRAGv2:
    """
    新版超图RAG实现类
    按照用户图片思路重新设计的超图RAG系统
    """
    
    def __init__(self, 
                 chunk_size: int = 500,
                 chunk_overlap: int = 50,
                 entity_model_path: str = None,
                 embedding_model_path: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        """
        初始化超图RAG系统
        
        Args:
            chunk_size: 文本块大小
            chunk_overlap: 文本块重叠大小
            entity_model_path: 实体识别模型路径
            embedding_model_path: 嵌入模型路径
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # 初始化嵌入模型
        try:
            self.embedding_model = SentenceTransformer(embedding_model_path)
            logger.info(f"成功加载嵌入模型: {embedding_model_path}")
        except Exception as e:
            logger.warning(f"无法加载嵌入模型: {e}, 将使用简单的文本匹配")
            self.embedding_model = None
        
        # 存储结构
        self.chunks: List[TextChunk] = []
        self.entities: Dict[str, Entity] = {}
        self.hyperedges: List[Hyperedge] = []
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        
        # 初始化jieba分词
        jieba.initialize()
        
    def text_chunking(self, text: str) -> List[TextChunk]:
        """
        文本切块功能
        将长文本分割成合适大小的块，每个块包含完整的句子
        
        Args:
            text: 输入文本
            
        Returns:
            List[TextChunk]: 文本块列表
        """
        logger.info("开始文本切块...")
        
        # 按句子分割文本
        sentences = self._split_into_sentences(text)
        chunks = []
        
        current_chunk = ""
        current_sentences = []
        chunk_id = 0
        
        for sentence in sentences:
            # 检查添加当前句子是否会超过块大小
            if len(current_chunk + sentence) > self.chunk_size and current_chunk:
                # 创建当前块
                chunk = TextChunk(
                    id=f"chunk_{chunk_id}",
                    content=current_chunk.strip(),
                    sentences=current_sentences.copy()
                )
                chunks.append(chunk)
                
                # 开始新块，保留重叠部分
                overlap_text = ""
                overlap_sentences = []
                current_length = 0
                
                # 从后往前添加句子作为重叠部分
                for i in range(len(current_sentences) - 1, -1, -1):
                    if current_length + len(current_sentences[i]) <= self.chunk_overlap:
                        overlap_text = current_sentences[i] + " " + overlap_text
                        overlap_sentences.insert(0, current_sentences[i])
                        current_length += len(current_sentences[i])
                    else:
                        break
                
                current_chunk = overlap_text + sentence
                current_sentences = overlap_sentences + [sentence]
                chunk_id += 1
            else:
                current_chunk += sentence
                current_sentences.append(sentence)
        
        # 添加最后一个块
        if current_chunk.strip():
            chunk = TextChunk(
                id=f"chunk_{chunk_id}",
                content=current_chunk.strip(),
                sentences=current_sentences
            )
            chunks.append(chunk)
        
        logger.info(f"文本切块完成，共生成 {len(chunks)} 个文本块")
        self.chunks = chunks
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """
        将文本分割成句子
        
        Args:
            text: 输入文本
            
        Returns:
            List[str]: 句子列表
        """
        # 中文句子分割规则
        sentence_endings = r'[。！？；\n]+'
        sentences = re.split(sentence_endings, text)
        
        # 清理空句子和过短句子
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]
        
        return sentences
    
    def extract_entities_from_chunks(self) -> Dict[str, List[Entity]]:
        """
        从所有文本块中提取实体
        对每个句子进行多实体提取
        
        Returns:
            Dict[str, List[Entity]]: 按块ID组织的实体列表
        """
        logger.info("开始从文本块中提取实体...")
        
        chunk_entities = {}
        
        for chunk in self.chunks:
            entities = []
            
            for sentence in chunk.sentences:
                # 对每个句子进行实体提取
                sentence_entities = self._extract_entities_from_sentence(sentence, chunk.id)
                entities.extend(sentence_entities)
            
            chunk.entities = entities
            chunk_entities[chunk.id] = entities
            
            # 更新全局实体字典
            for entity in entities:
                if entity.name not in self.entities:
                    self.entities[entity.name] = entity
                else:
                    # 合并实体信息
                    existing = self.entities[entity.name]
                    if entity.description and not existing.description:
                        existing.description = entity.description
        
        logger.info(f"实体提取完成，共提取 {len(self.entities)} 个唯一实体")
        return chunk_entities
    
    def _extract_entities_from_sentence(self, sentence: str, chunk_id: str) -> List[Entity]:
        """
        从单个句子中提取实体
        使用jieba分词和词性标注进行实体识别
        
        Args:
            sentence: 输入句子
            chunk_id: 所属文本块ID
            
        Returns:
            List[Entity]: 实体列表
        """
        entities = []
        
        # 使用jieba进行词性标注
        words = pseg.cut(sentence)
        
        # 定义实体类型映射
        entity_type_mapping = {
            'n': '名词',      # 一般名词
            'nr': '人名',     # 人名
            'ns': '地名',     # 地名
            'nt': '机构名',   # 机构名
            'nz': '其他专名', # 其他专名
            'v': '动词',      # 动词
            'a': '形容词',    # 形容词
            'i': '成语',      # 成语
            'l': '习用语',    # 习用语
            'j': '简称',      # 简称
        }
        
        for word, flag in words:
            # 过滤条件：长度大于1，且是我们关心的词性
            if len(word) > 1 and flag in entity_type_mapping:
                entity = Entity(
                    name=word.strip(),
                    entity_type=entity_type_mapping.get(flag, '未知'),
                    source_sentence=sentence,
                    chunk_id=chunk_id,
                    description=f"从句子中提取的{entity_type_mapping.get(flag, '未知')}"
                )
                entities.append(entity)
        
        return entities
    
    def build_hyperedges_from_chunks(self) -> List[Hyperedge]:
        """
        从文本块构建超边
        通过实体聚合获得相应的超边
        
        Returns:
            List[Hyperedge]: 超边列表
        """
        logger.info("开始构建超边...")
        
        hyperedges = []
        
        for chunk in self.chunks:
            # 为每个文本块构建超边
            chunk_hyperedges = self._build_hyperedges_for_chunk(chunk)
            hyperedges.extend(chunk_hyperedges)
            chunk.hyperedges = chunk_hyperedges
        
        self.hyperedges = hyperedges
        logger.info(f"超边构建完成，共构建 {len(hyperedges)} 个超边")
        return hyperedges
    
    def _build_hyperedges_for_chunk(self, chunk: TextChunk) -> List[Hyperedge]:
        """
        为单个文本块构建超边
        
        Args:
            chunk: 文本块
            
        Returns:
            List[Hyperedge]: 超边列表
        """
        hyperedges = []
        
        # 按句子构建超边
        for sentence in chunk.sentences:
            sentence_entities = [e for e in chunk.entities if e.source_sentence == sentence]
            
            if len(sentence_entities) >= 2:
                # 句子内实体共现超边
                entity_names = {e.name for e in sentence_entities}
                hyperedge = Hyperedge(
                    entities=entity_names,
                    weight=1.0,
                    description=f"句子内实体共现: {sentence[:50]}...",
                    source_chunk=chunk.id,
                    edge_type="sentence_cooccurrence"
                )
                hyperedges.append(hyperedge)
        
        # 构建语义相关超边
        semantic_hyperedges = self._build_semantic_hyperedges(chunk)
        hyperedges.extend(semantic_hyperedges)
        
        return hyperedges
    
    def _build_semantic_hyperedges(self, chunk: TextChunk) -> List[Hyperedge]:
        """
        构建语义相关的超边
        
        Args:
            chunk: 文本块
            
        Returns:
            List[Hyperedge]: 语义超边列表
        """
        hyperedges = []
        
        # 按实体类型分组
        entity_groups = defaultdict(list)
        for entity in chunk.entities:
            entity_groups[entity.entity_type].append(entity)
        
        # 为每个实体类型组构建超边
        for entity_type, entities in entity_groups.items():
            if len(entities) >= 2:
                entity_names = {e.name for e in entities}
                hyperedge = Hyperedge(
                    entities=entity_names,
                    weight=0.8,
                    description=f"{entity_type}类实体语义关联",
                    source_chunk=chunk.id,
                    edge_type="semantic_relation"
                )
                hyperedges.append(hyperedge)
        
        return hyperedges
    
    def extract_entities_from_query(self, query: str) -> List[Entity]:
        """
        从查询问题中提取实体
        
        Args:
            query: 查询问题
            
        Returns:
            List[Entity]: 问题中的实体列表
        """
        logger.info(f"从查询中提取实体: {query}")
        
        # 将查询作为单个句子处理
        entities = self._extract_entities_from_sentence(query, "query")
        
        logger.info(f"从查询中提取到 {len(entities)} 个实体: {[e.name for e in entities]}")
        return entities
    
    def build_query_hyperedges(self, query_entities: List[Entity]) -> List[Hyperedge]:
        """
        为查询实体构建超边
        
        Args:
            query_entities: 查询实体列表
            
        Returns:
            List[Hyperedge]: 查询超边列表
        """
        logger.info("为查询构建超边...")
        
        hyperedges = []
        
        if len(query_entities) >= 2:
            # 查询实体共现超边
            entity_names = {e.name for e in query_entities}
            hyperedge = Hyperedge(
                entities=entity_names,
                weight=1.0,
                description="查询实体共现",
                source_chunk="query",
                edge_type="query_cooccurrence"
            )
            hyperedges.append(hyperedge)
        
        # 按实体类型分组构建语义超边
        entity_groups = defaultdict(list)
        for entity in query_entities:
            entity_groups[entity.entity_type].append(entity)
        
        for entity_type, entities in entity_groups.items():
            if len(entities) >= 2:
                entity_names = {e.name for e in entities}
                hyperedge = Hyperedge(
                    entities=entity_names,
                    weight=0.9,
                    description=f"查询{entity_type}类实体关联",
                    source_chunk="query",
                    edge_type="query_semantic"
                )
                hyperedges.append(hyperedge)
        
        logger.info(f"为查询构建了 {len(hyperedges)} 个超边")
        return hyperedges
    
    def match_hyperedges(self, query_hyperedges: List[Hyperedge], top_k: int = 5) -> List[Dict]:
        """
        使用查询超边匹配文档超边
        
        Args:
            query_hyperedges: 查询超边列表
            top_k: 返回前k个最相关的结果
            
        Returns:
            List[Dict]: 匹配结果列表
        """
        logger.info("开始超边匹配...")
        
        matches = []
        
        for doc_hyperedge in self.hyperedges:
            max_similarity = 0.0
            best_query_edge = None
            
            for query_hyperedge in query_hyperedges:
                similarity = self._calculate_hyperedge_similarity(query_hyperedge, doc_hyperedge)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_query_edge = query_hyperedge
            
            if max_similarity > 0:
                matches.append({
                    'doc_hyperedge': doc_hyperedge,
                    'query_hyperedge': best_query_edge,
                    'similarity': max_similarity,
                    'chunk_id': doc_hyperedge.source_chunk
                })
        
        # 按相似度排序
        matches.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"找到 {len(matches)} 个匹配，返回前 {top_k} 个")
        return matches[:top_k]
    
    def _calculate_hyperedge_similarity(self, edge1: Hyperedge, edge2: Hyperedge) -> float:
        """
        计算两个超边的相似度
        
        Args:
            edge1: 超边1
            edge2: 超边2
            
        Returns:
            float: 相似度分数 (0-1)
        """
        # 实体交集相似度
        intersection = edge1.entities & edge2.entities
        union = edge1.entities | edge2.entities
        
        if not union:
            return 0.0
        
        jaccard_similarity = len(intersection) / len(union)
        
        # 如果有嵌入模型，计算语义相似度
        if self.embedding_model:
            try:
                edge1_text = " ".join(edge1.entities) + " " + edge1.description
                edge2_text = " ".join(edge2.entities) + " " + edge2.description
                
                embeddings = self.embedding_model.encode([edge1_text, edge2_text])
                semantic_similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
                
                # 综合相似度
                combined_similarity = 0.6 * jaccard_similarity + 0.4 * semantic_similarity
                return float(combined_similarity)
            except Exception as e:
                logger.warning(f"计算语义相似度时出错: {e}")
                return jaccard_similarity
        
        return jaccard_similarity
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        检索相关文本块的主要接口
        
        Args:
            query: 查询问题
            top_k: 返回前k个最相关的结果
            
        Returns:
            List[Dict]: 检索结果
        """
        logger.info(f"开始检索相关文本块，查询: {query}")
        
        # 1. 从查询中提取实体
        query_entities = self.extract_entities_from_query(query)
        
        # 2. 为查询实体构建超边
        query_hyperedges = self.build_query_hyperedges(query_entities)
        
        # 3. 匹配超边
        matches = self.match_hyperedges(query_hyperedges, top_k * 2)  # 获取更多候选
        
        # 4. 获取相关文本块
        chunk_scores = defaultdict(float)
        chunk_details = {}
        
        for match in matches:
            chunk_id = match['chunk_id']
            similarity = match['similarity']
            
            chunk_scores[chunk_id] += similarity
            
            if chunk_id not in chunk_details:
                chunk = next((c for c in self.chunks if c.id == chunk_id), None)
                if chunk:
                    chunk_details[chunk_id] = {
                        'chunk': chunk,
                        'matches': []
                    }
            
            if chunk_id in chunk_details:
                chunk_details[chunk_id]['matches'].append(match)
        
        # 5. 排序并返回结果
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            if chunk_id in chunk_details:
                results.append({
                    'chunk_id': chunk_id,
                    'content': chunk_details[chunk_id]['chunk'].content,
                    'score': score,
                    'matches': chunk_details[chunk_id]['matches'],
                    'entities': [e.name for e in chunk_details[chunk_id]['chunk'].entities]
                })
        
        logger.info(f"检索完成，返回 {len(results)} 个相关文本块")
        return results
    
    def save_hypergraph(self, filepath: str):
        """
        保存超图结构到文件
        
        Args:
            filepath: 保存路径
        """
        logger.info(f"保存超图到: {filepath}")
        
        data = {
            'chunks': [
                {
                    'id': chunk.id,
                    'content': chunk.content,
                    'sentences': chunk.sentences,
                    'entities': [
                        {
                            'name': e.name,
                            'type': e.entity_type,
                            'description': e.description,
                            'source_sentence': e.source_sentence
                        } for e in chunk.entities
                    ],
                    'hyperedges': [
                        {
                            'entities': list(he.entities),
                            'weight': he.weight,
                            'description': he.description,
                            'edge_type': he.edge_type
                        } for he in chunk.hyperedges
                    ]
                } for chunk in self.chunks
            ],
            'entities': [
                {
                    'name': e.name,
                    'type': e.entity_type,
                    'description': e.description,
                    'source_sentence': e.source_sentence,
                    'chunk_id': e.chunk_id
                } for e in self.entities.values()
            ],
            'hyperedges': [
                {
                    'entities': list(he.entities),
                    'weight': he.weight,
                    'description': he.description,
                    'source_chunk': he.source_chunk,
                    'edge_type': he.edge_type
                } for he in self.hyperedges
            ]
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info("超图保存完成")


def main():
    """
    主函数 - 演示超图RAG的使用
    """
    # 示例文本
    sample_text = """
    人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器。
    机器学习是人工智能的一个重要分支，通过算法使机器能够从数据中学习并做出决策或预测。
    深度学习是机器学习的一个子领域，它基于人工神经网络，特别是深层神经网络来学习数据的表示。
    自然语言处理是人工智能的另一个重要分支，它研究如何让计算机理解和生成人类语言。
    计算机视觉致力于让机器能够识别和理解图像和视频中的内容。
    """
    
    # 初始化超图RAG系统
    rag_system = HypergraphRAGv2(chunk_size=200, chunk_overlap=50)
    
    print("=== 超图RAG系统演示 ===")
    
    # 1. 文本切块
    print("\n1. 文本切块...")
    chunks = rag_system.text_chunking(sample_text)
    for i, chunk in enumerate(chunks):
        print(f"块 {i}: {chunk.content[:100]}...")
    
    # 2. 实体提取
    print("\n2. 实体提取...")
    chunk_entities = rag_system.extract_entities_from_chunks()
    for chunk_id, entities in chunk_entities.items():
        print(f"{chunk_id}: {[e.name for e in entities]}")
    
    # 3. 构建超边
    print("\n3. 构建超边...")
    hyperedges = rag_system.build_hyperedges_from_chunks()
    for i, edge in enumerate(hyperedges[:5]):  # 只显示前5个
        print(f"超边 {i}: {edge.entities} - {edge.description}")
    
    # 4. 查询检索
    print("\n4. 查询检索...")
    query = "什么是机器学习？"
    results = rag_system.retrieve_relevant_chunks(query, top_k=2)
    
    for i, result in enumerate(results):
        print(f"\n结果 {i+1} (相似度: {result['score']:.3f}):")
        print(f"内容: {result['content'][:150]}...")
        print(f"相关实体: {result['entities']}")
    
    # 5. 保存超图
    print("\n5. 保存超图...")
    rag_system.save_hypergraph("hypergraph_structure.json")
    print("超图结构已保存到 hypergraph_structure.json")


if __name__ == "__main__":
    main()
