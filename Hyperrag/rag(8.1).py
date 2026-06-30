"""
HyperRAG v8.1: Zero-shot Graph Attention Cascading Framework
(动态自注意力架构 + 完整功能版)

Architecture:
- L0: Fuzzy Overlapping Topic Manifold
- L1: Macro Chunk Layer (Topic + Sentences_Agg + Raw)
- L2: Meso Sentence Layer (Chunk_New + Entities_Agg + Raw)
- L3: Micro Entity Layer (Sentence_New + Raw)

Pipeline:
1. Indexing: 严格级联 Topic → Chunk → Sentence → Entity
2. Retrieval: 对称级联 Query_Chunk → Query_Sentence → Query_Entity

核心特性 (v8.0 算法 + v7.3 功能):
- 动态自注意力融合 (attention_fusion): 替代所有硬编码权重
- 动态多粒度打分 (attention_scoring): 替代固定 0.3/0.4/0.3 公式
- 完整缓存系统、Markdown文件加载、多轮实体提取、碎片过滤
"""

import asyncio
import json
import re
import uuid
import numpy as np
import os
import sys
import pickle
import hashlib
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set

# --- Dependency Check ---
try:
    import igraph as ig
    import leidenalg
    LEIDEN_AVAILABLE = True
except ImportError:
    LEIDEN_AVAILABLE = False
    print("[System] Warning: 'leidenalg' or 'igraph' not found. Fallback to greedy clustering.")

# ============================================================================
# 0. Global Configuration & API (使用 hyperrag 封装)
# ============================================================================

sys.path.append("./Hyper-RAG")
from hyperrag.llm import openai_complete_if_cache, openai_embedding
from hyperrag.operate import chunking_by_token_size
from experiment_config import LLM_MODEL, LLM_BASE_URL, LLM_API_KEY, EMB_MODEL, EMB_BASE_URL, EMB_API_KEY

os.environ["OPENAI_API_KEY"] = os.getenv("SILICONFLOW_API_KEY", "")

def normalize(v: np.ndarray) -> np.ndarray:
    """L2 Normalization to project vectors onto the unit hypersphere."""
    norm = np.linalg.norm(v)
    return v / norm if norm > 1e-9 else v

def fix_json_escapes(json_str: str) -> str:
    """修复 LLM 返回的 JSON 中无效的转义字符"""
    result = []
    i = 0
    valid_escapes = {'n', 't', 'r', 'b', 'f', '"', '\\', '/', 'u'}
    while i < len(json_str):
        if json_str[i] == '\\' and i + 1 < len(json_str):
            next_char = json_str[i + 1]
            if next_char in valid_escapes:
                result.append(json_str[i:i+2])
                i += 2
            elif next_char == '\\':
                result.append('\\\\')
                i += 2
            else:
                result.append('\\\\')
                result.append(next_char)
                i += 2
        else:
            result.append(json_str[i])
            i += 1
    return ''.join(result)

# --- Real Embedding API ---
async def get_embedding_api(texts: List[str]) -> List[np.ndarray]:
    """调用真实 Embedding API (自动分批避免超出最大 batch size限制)"""
    if not texts: 
        return []
    api_key = EMB_API_KEY or os.environ.get("OPENAI_API_KEY")
    
    batch_size = 50
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = await openai_embedding(
            texts=batch_texts,
            model=EMB_MODEL,
            base_url=EMB_BASE_URL,
            api_key=api_key,
        )
        all_embeddings.extend(batch_embeddings)
        
    return [np.array(e) for e in all_embeddings]

# --- Real Completion API ---
async def get_completion_api(prompt: str) -> str:
    """调用真实 LLM API"""
    return await openai_complete_if_cache(
        model=LLM_MODEL,
        prompt=prompt,
        base_url=LLM_BASE_URL,
        api_key=LLM_API_KEY
    )

# ============================================================================
# 0.5 [Attention] 核心动态注意力算子 (v8.0 核心)
# ============================================================================

def attention_fusion(query_vec: np.ndarray, key_vecs: List[np.ndarray], temperature: float = 0.1) -> np.ndarray:
    """
    算子 1: 向量融合交叉注意力
    通过计算 query_vec 与各个 key_vec 的余弦相似度，利用 Softmax 动态分配融合权重。
    """
    if not key_vecs: return query_vec
    
    sims = np.array([np.dot(query_vec, k) for k in key_vecs])
    logits = sims / temperature
    exp_logits = np.exp(logits - np.max(logits))
    weights = exp_logits / (np.sum(exp_logits) + 1e-9)
    
    fused_vec = np.zeros_like(query_vec)
    for w, k in zip(weights, key_vecs):
        fused_vec += w * k
        
    return normalize(fused_vec)

def attention_scoring(q_vec: np.ndarray, doc_c: np.ndarray, doc_s: np.ndarray, score_ent: float, temperature: float = 0.08) -> Tuple[float, Dict[str, float]]:
    """
    算子 2: 动态多粒度打分注意力
    检索时，根据 Query 的意图自动分配宏观(Chunk)与微观(Entity)的打分权重。
    """
    sim_c = np.dot(q_vec, doc_c)
    sim_s = np.dot(q_vec, doc_s)
    sim_e = score_ent
    
    logits = np.array([sim_c, sim_s, sim_e]) / temperature
    exp_logits = np.exp(logits - np.max(logits))
    weights = exp_logits / (np.sum(exp_logits) + 1e-9)
    
    final_score = float(np.sum(weights * np.array([sim_c, sim_s, sim_e])))
    
    return final_score, {"w_c": float(weights[0]), "w_s": float(weights[1]), "w_e": float(weights[2])}

# ============================================================================
# 1. Data Topology (Cascading State)
# ============================================================================

@dataclass
class EntityNode:
    """[L3] Micro Node - 级联状态"""
    name: str
    weight: float = 1.0
    raw_embedding: Optional[np.ndarray] = None    # 原始向量
    final_embedding: Optional[np.ndarray] = None  # 级联更新后的向量

@dataclass
class SentenceNode:
    """[L2] Meso Node - 级联状态 (原 HyperEdge)"""
    sent_id: str
    chunk_ids: List[int] = field(default_factory=list)  # 支持多对多：一个句子可关联多个 chunk
    text: str = ""
    entities: List[EntityNode] = field(default_factory=list)
    
    raw_embedding: Optional[np.ndarray] = None      # 原始向量
    ent_agg_embedding: Optional[np.ndarray] = None  # 实体聚合向量 (Bottom-Up)
    final_embedding: Optional[np.ndarray] = None    # 级联更新后的向量

@dataclass
class ChunkNode:
    """[L1] Macro Node - 级联状态"""
    chunk_id: int
    text: str
    
    raw_embedding: Optional[np.ndarray] = None       # 原始向量
    sent_agg_embedding: Optional[np.ndarray] = None  # 句子聚合向量 (Bottom-Up)
    final_embedding: Optional[np.ndarray] = None     # 级联更新后的向量
    
    sentence_ids: List[str] = field(default_factory=list)
    topic_memberships: Dict[int, float] = field(default_factory=dict)

@dataclass
class TopicNode:
    """[L0] Latent Node (Manifold Anchor)"""
    topic_id: int
    topic_vector: np.ndarray  # Centroid (mu_k)
    description: str = "Latent Topic"

# ============================================================================
# 2. NLP Processors (详细 Prompt 版 - 来自 v7.3)
# ============================================================================

SENTENCE_EXTRACTION_PROMPT = """
请从以下段落中提取所有句子，并标注每个句子在原文中的起始和结束位置（字符索引，从0开始）。
段落：{paragraph}
请以JSON格式输出：{{ "sentences": [ {{"text": "...", "start_pos": 0, "end_pos": 10}}, ... ] }}
"""

ENTITY_EXTRACTION_PROMPT = """
请从以下句子中提取所有关键信息单元（实体及核心概念），并标注重要性权重。

句子：{sentence}

请务必提取以下两类信息：
1. **具体实体**：人名、地名、组织、物品、具体时间点、数值。
2. **抽象概念/意图**：
   - 核心动作（如"出使"、"争执"、"截获"）
   - 目的/任务（如"任务"、"意图"、"为了..."）
   - 结果/后果（如"结局"、"下场"、"导致"）
   - 原因/理由（如"因为"、"缘由"）
   - 方式/手段（如"通过..."、"利用..."）
   - 情感/态度（如"愤怒"、"怀疑"、"敬佩"）

【重要性权重 (importance) 打分标准】：
- **1.0 (绝对核心)**: 句子的主语、核心谓语动词、用户查询的最主要意图。缺失该词会导致句子无法理解。
- **0.7 ~ 0.9 (关键信息)**: 核心宾语、必要的限定条件、关键的地点或时间。
- **0.4 ~ 0.6 (次要细节)**: 形容词修饰、副词（程度/方式）、非关键的背景物品、单纯的环境描写。
- **0.1 ~ 0.3 (背景噪音)**: 通用的量词、虚词代指、口语化的连接词、无关紧要的景色描写。

示例输入 1 (复杂叙事句)：
"虽然夜色已深，狂风呼啸，但霍云雷仍坚持在营帐内仔细擦拭那柄家传的宝剑，以备明日与匈奴的决战。"
示例输出 1：
{{"entities": [
    {{"name": "霍云雷", "type": "人物", "importance": 1.0, "description": "动作执行者"}},
    {{"name": "擦拭", "type": "动作", "importance": 0.9, "description": "核心动作"}},
    {{"name": "宝剑", "type": "物品", "importance": 1.0, "description": "动作承受对象"}},
    {{"name": "决战", "type": "目的", "importance": 0.9, "description": "行动的目的"}},
    {{"name": "匈奴", "type": "组织", "importance": 0.8, "description": "敌对目标"}},
    {{"name": "家传的", "type": "修饰", "importance": 0.5, "description": "物品属性细节"}},
    {{"name": "仔细", "type": "修饰", "importance": 0.4, "description": "动作方式"}},
    {{"name": "营帐内", "type": "地点", "importance": 0.3, "description": "背景地点"}},
    {{"name": "夜色已深", "type": "环境", "importance": 0.2, "description": "环境背景"}},
    {{"name": "狂风呼啸", "type": "环境", "importance": 0.2, "description": "环境背景"}}
]}}

示例输入 2 (复杂查询句)：
"在于阗国发生政变之前，李长风采取了哪些具体的经济手段来稳定民心？"
示例输出 2：
{{"entities": [
    {{"name": "经济手段", "type": "意图", "importance": 1.0, "description": "查询的核心目标"}},
    {{"name": "李长风", "type": "人物", "importance": 1.0, "description": "行为主体"}},
    {{"name": "稳定民心", "type": "目的", "importance": 0.9, "description": "行为目的"}},
    {{"name": "于阗国", "type": "地点", "importance": 0.8, "description": "事件发生地"}},
    {{"name": "政变", "type": "事件", "importance": 0.7, "description": "时间背景事件"}},
    {{"name": "具体的", "type": "修饰", "importance": 0.4, "description": "程度修饰"}},
    {{"name": "之前", "type": "时间", "importance": 0.3, "description": "时间关系词"}}
]}}

示例输入 3 (长难句：包含因果、手段与结果的复合逻辑)：
"鉴于边境连年战乱导致国库空虚，宰相王安石力排众议，坚持推行'青苗法'，试图通过政府低息贷款来抑制豪强兼并，从而增加国家财政收入。"
示例输出 3：
{{"entities": [
    {{"name": "王安石", "type": "人物", "importance": 1.0, "description": "变法主导者"}},
    {{"name": "推行青苗法", "type": "动作", "importance": 1.0, "description": "核心政治动作"}},
    {{"name": "增加国家财政收入", "type": "目的", "importance": 1.0, "description": "最终战略目标"}},
    {{"name": "抑制豪强兼并", "type": "目的", "importance": 0.9, "description": "直接战术目标"}},
    {{"name": "政府低息贷款", "type": "手段", "importance": 0.9, "description": "具体实施手段"}},
    {{"name": "国库空虚", "type": "原因", "importance": 0.8, "description": "变法的背景原因"}},
    {{"name": "边境连年战乱", "type": "背景", "importance": 0.6, "description": "外部环境"}},
    {{"name": "力排众议", "type": "修饰", "importance": 0.5, "description": "体现人物决心"}},
    {{"name": "鉴于", "type": "虚词", "importance": 0.2, "description": "逻辑连接词"}}
]}}

示例输入 4 (多实体交互：包含工具、证据与推论)：
"霍云雷利用截获的匈奴鹰信，模仿左贤王的笔迹伪造了一份撤军手令，成功诱骗潜伏在城内的敌方间谍暴露了行踪。"
示例输出 4：
{{"entities": [
    {{"name": "霍云雷", "type": "人物", "importance": 1.0, "description": "计谋执行者"}},
    {{"name": "诱骗间谍暴露", "type": "结果", "importance": 1.0, "description": "行动的直接战果"}},
    {{"name": "伪造撤军手令", "type": "动作", "importance": 0.9, "description": "核心欺骗手段"}},
    {{"name": "模仿笔迹", "type": "手段", "importance": 0.8, "description": "技术手段"}},
    {{"name": "匈奴鹰信", "type": "物品", "importance": 0.7, "description": "利用的道具"}},
    {{"name": "敌方间谍", "type": "人物", "importance": 0.9, "description": "受骗对象"}},
    {{"name": "左贤王", "type": "人物", "importance": 0.6, "description": "被冒充的对象"}},
    {{"name": "潜伏在城内的", "type": "修饰", "importance": 0.4, "description": "对象的位置状态"}}
]}}

请以JSON格式输出，只输出一个JSON对象，不要有任何注释或额外文字：
{{"entities": [...]}}
"""

ENTITY_SUPPLEMENT_PROMPT = """
请检查以下句子，看看是否还有遗漏的实体没有被提取。

句子：{sentence}

已提取的实体：{existing_entities}

请仔细检查是否遗漏了：
- 人物（包括代词指代的人物、群体称呼如"众人"、"丫鬟们"）
- 地点/场所
- 物品/器具
- 动作/事件
- 时间表达
- 情感/状态

如果有遗漏，请补充提取。如果没有遗漏，返回空列表。

示例输出（有遗漏时）：
{{"entities": [{{"name": "众人", "type": "群体", "importance": 0.5, "description": "在场的人们"}}]}}

示例输出（无遗漏时）：
{{"entities": []}}

只输出一个JSON对象，不要有任何注释或额外文字：
"""

QUERY_EXPAND_PROMPT = """
你是一个搜索查询优化器。
请将用户的查询扩展为一个更详细的问句，包含更多相关的关键词、上下文和可能涉及的实体。

重要规则：
1. 如果原查询是问句，扩展后必须保持问句形式
2. 不要回答问题，只是扩展问题本身
3. 保留原查询的核心疑问词（什么、如何、为什么等）

示例输入："李长风的最终结局是什么？"
示例输出："作为出使西域的使节，李长风在经历了各种艰险和外交博弈之后，他的最终命运和结局是什么？他是否完成了使命并安全返回？"

用户查询：{query}
请直接输出扩展后的问句，不要有任何解释：
"""

# ============================================================================
# 3. 文本处理函数 (句子分割 + 多轮实体提取 - 来自 v7.3)
# ============================================================================

async def extract_sentences(paragraph: str) -> List[str]:
    """从段落中提取句子"""
    prompt = SENTENCE_EXTRACTION_PROMPT.format(paragraph=paragraph)
    response = await get_completion_api(prompt)
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        json_str = json_match.group() if json_match else response
        
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        json_str = re.sub(r'}\s*{', '},{', json_str)
        json_str = re.sub(r'"\s*\n\s*"', '","', json_str)
        json_str = fix_json_escapes(json_str)
        
        data = json.loads(json_str)
        sentences_data = data.get("sentences", [])
        result = []
        for s in sentences_data:
            text = s.get("text", s) if isinstance(s, dict) else s
            text = str(text).strip()
            if len(text) > 10 and not re.match(r'^[。，、；：！？""''「」【】（）\s]', text) and not re.match(r'^(的|了|着|过|在|和|与|或)', text):
                result.append(text)
        return result
    except Exception as e:
        print(f"[Warning] 句子提取错误: {e}, 使用降级分割")
        sentences = re.split(r'[。！？；\n]+', paragraph)
        result = []
        for s in sentences:
            s = s.strip()
            if len(s) > 10 and not re.match(r'^[。，、；：！？""''「」【】（）\s]', s) and not re.match(r'^(的|了|着|过|在|和|与|或)', s):
                result.append(s)
        return result

async def extract_entities_multi_round(sentence_text: str, max_rounds: int = 4) -> List[EntityNode]:
    """多轮实体提取：第一轮基础提取 + 后续轮次补充遗漏"""
    all_entities = []
    existing_names = set()
    
    # 第一轮：基础提取
    prompt = ENTITY_EXTRACTION_PROMPT.format(sentence=sentence_text)
    response = await get_completion_api(prompt)
    try:
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        json_str = fix_json_escapes(json_match.group() if json_match else response)
        data = json.loads(json_str)
        for ent_data in data.get("entities", []):
            entity = EntityNode(
                name=ent_data["name"],
                weight=float(ent_data.get("importance", 0.5))
            )
            if entity.name not in existing_names:
                all_entities.append(entity)
                existing_names.add(entity.name)
    except Exception as e:
        print(f"[Warning] 实体提取错误(第1轮): {e}")
    
    # 后续轮次：补充提取
    for round_num in range(2, max_rounds + 1):
        if not all_entities:
            break
            
        existing_list = ", ".join(existing_names)
        prompt = ENTITY_SUPPLEMENT_PROMPT.format(
            sentence=sentence_text,
            existing_entities=existing_list
        )
        response = await get_completion_api(prompt)
        
        try:
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            json_str = fix_json_escapes(json_match.group() if json_match else response)
            data = json.loads(json_str)
            new_entities = data.get("entities", [])
            
            if not new_entities:
                break
                
            for ent_data in new_entities:
                if not isinstance(ent_data, dict):
                    continue
                name = ent_data.get("name", "")
                if name and name not in existing_names:
                    entity = EntityNode(
                        name=name,
                        weight=float(ent_data.get("importance", 0.5))
                    )
                    all_entities.append(entity)
                    existing_names.add(name)
                    print(f"    [补充] 发现遗漏实体: {name}")
        except Exception as e:
            print(f"[Warning] 实体提取错误(第{round_num}轮): {e}")
            break
    
    return all_entities

# ============================================================================
# 4. 文档加载与切块模块 (来自 v7.3)
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
    """清理 Markdown 文本，移除图片链接、URL等"""
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
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
# 5. 缓存工具函数 (来自 v7.3)
# ============================================================================

CACHE_DIR = "./hyperedge_cache"

def get_cache_path(content: str, prefix: str = "hyperrag_v81", model: str = None) -> str:
    """根据内容和模型生成缓存文件路径（不同模型不同缓存）"""
    os.makedirs(CACHE_DIR, exist_ok=True)
    content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
    if model is None:
        model = LLM_MODEL
    # 模型名中的特殊字符替换为下划线
    safe_model = re.sub(r'[^\w\-.]', '_', model)
    return os.path.join(CACHE_DIR, f"{prefix}_{safe_model}_{content_hash}.pkl")

def save_engine_state(engine, cache_path: str):
    """保存引擎状态到缓存文件"""
    state = {
        "chunks": engine.chunks,
        "sentences": engine.sentences,
        "topics": engine.topics,
        "global_registry": engine.global_registry
    }
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    tmp_path = f"{cache_path}.tmp.{os.getpid()}"
    try:
        with open(tmp_path, 'wb') as f:
            pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, cache_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise
    print(f"[Cache] 已保存引擎状态到 {cache_path}")

def load_engine_state(engine, cache_path: str) -> bool:
    """从缓存文件加载引擎状态"""
    if os.path.exists(cache_path):
        if os.path.getsize(cache_path) == 0:
            print(f"[Cache] Warning: empty cache ignored: {cache_path}")
            return False
        try:
            with open(cache_path, 'rb') as f:
                state = pickle.load(f)
        except (EOFError, pickle.UnpicklingError, OSError, KeyError, TypeError) as e:
            print(f"[Cache] Warning: invalid cache ignored: {cache_path} ({e})")
            return False
        required_keys = {"chunks", "sentences", "topics", "global_registry"}
        if not isinstance(state, dict) or not required_keys.issubset(state):
            print(f"[Cache] Warning: invalid cache schema ignored: {cache_path}")
            return False
        engine.chunks = state["chunks"]
        engine.sentences = state["sentences"]
        engine.topics = state["topics"]
        engine.global_registry = state["global_registry"]
        print(f"[Cache] 从 {cache_path} 加载了引擎状态")
        print(f"  - Topics: {len(engine.topics)}, Chunks: {len(engine.chunks)}, Sentences: {len(engine.sentences)}")
        return True
    return False

# ============================================================================
# 6. 实体一致性校验器 (防止张冠李戴 - 来自 v7.3)
# ============================================================================

VERIFY_PROMPT = """你是一个客观严谨的知识库事实校验员。请判断以下【检索片段】是否对回答【用户问题】有任何直接或间接的帮助。

用户问题：{query}
检索片段：{text}

【校验核心原则】：
1. **主体一致性（一票否决）**：如果片段明确描述的是实体B，而用户询问的是实体A（张冠李戴），或者讨论的场景完全不匹配，请判定为 false。
2. **碎片拼图包容性（宽容放行）**：综合知识库中的答案往往散落在多处。如果该片段只提供了一个局部线索（如：一个前置概念、一个基础定义、一段历史背景、一个操作步骤的中间环节、一个公式），只要它与用户问题探讨的是**同一个核心主题**，就应当保留，判定为 true。千万不要因为"单看这一段无法完整作答全貌"而拒绝它。
3. **无关噪音**：如果片段完全是毫无逻辑关联的通用废话或风马牛不相及的内容，判定为 false。

请以JSON格式输出：
{{"is_valid": true/false, "reason": "简短的判断理由"}}
"""

class ConsistencyVerifier:
    """专门用于清洗检索结果，解决'张冠李戴'问题"""
    
    @staticmethod
    async def verify_single(query: str, res: dict) -> Tuple[bool, str]:
        """校验单条结果"""
        # 【修改点】：优先获取预先扩展好的上下文 expanded_text
        text = res.get('expanded_text', res.get('text', res.get('sentence', '')))
        prompt = VERIFY_PROMPT.format(query=query, text=text)
        
        try:
            response = await get_completion_api(prompt)
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = fix_json_escapes(json_match.group())
                data = json.loads(json_str)
                is_valid = data.get("is_valid", False)
                reason = data.get("reason", "无理由")
                return is_valid, reason
            return True, "解析失败，默认保留"
        except Exception as e:
            return True, f"校验异常: {e}"
    
    @staticmethod
    async def verify_batch(query: str, results: List[Dict], concurrency_limit: int = 5) -> List[Dict]:
        """并发校验一批结果"""
        if not results:
            return []
            
        print(f"\n[Verifier] 开始校验 {len(results)} 条检索结果...")
        sem = asyncio.Semaphore(concurrency_limit)
        
        async def _check_with_sem(res):
            async with sem:
                is_valid, reason = await ConsistencyVerifier.verify_single(query, res)
                return res, is_valid, reason
        
        tasks = [_check_with_sem(res) for res in results]
        checked_data = await asyncio.gather(*tasks)
        
        valid_results = []
        for res, is_valid, reason in checked_data:
            if is_valid:
                res["verify_reason"] = reason
                valid_results.append(res)
            else:
                score = res.get('score', 0)
                text_preview = res.get('text', res.get('sentence', ''))[:50]
                print(f"  [剔除] (Score {score:.2f}): {reason} | {text_preview}...")
        
        print(f"[Verifier] 校验完成: 保留 {len(valid_results)}/{len(results)}")
        return valid_results

# ============================================================================
# 7. HyperRAG v8.1 Cascading Engine (v8.0 Attention + v7.3 Structure)
# ============================================================================

class HyperRAG_Attention_v81:
    """
    完全对称的级联架构 + 动态自注意力
    
    索引端级联：Topic → Chunk → Sentence → Entity (使用 attention_fusion)
    检索端级联：Query_Chunk → Query_Sentence → Query_Entity (使用 attention_fusion)
    打分：attention_scoring 动态分配 Chunk/Sentence/Entity 权重
    """
    
    def __init__(self):
        self.chunks: Dict[int, ChunkNode] = {}
        self.sentences: Dict[str, SentenceNode] = {}
        self.topics: Dict[int, TopicNode] = {}
        self.global_registry: Dict[str, List[str]] = {}  # Entity Name -> List[SentenceIDs]

    # =========================================================================
    # Phase 1: Index Construction (Strictly Top-Down Cascade)
    # =========================================================================
    
    async def build_index(self, raw_text: str, max_token_size=400, overlap_token_size=50, overlap_threshold=0.6):
        """
        构建索引 - 严格级联流程
        
        Phase A: 准备原始数据 (Raw Prep)
        Phase B: 主题发现 (Topic Discovery)
        Phase C: 注意力级联更新 (Attention Cascading Update: Chunk → Sentence → Entity)
        """
        print(f"\n{'='*60}")
        print("[Build] HyperRAG v8.1 Attention Cascading Indexing Pipeline")
        print(f"{'='*60}")

        # === Phase A: Preparation (Raw & Extraction) ===
        print(f"\n[Phase A] Raw Data Preparation...")
        
        # A1. Token级切块 & Chunk Raw Embedding
        chunk_data = chunk_document(raw_text, max_token_size=max_token_size, overlap_token_size=overlap_token_size)
        chunk_texts = [c["content"] for c in chunk_data]
        print(f"  [A1] Chunking: {len(chunk_texts)} chunks")
        
        chunk_vecs = await get_embedding_api(chunk_texts)
        for i, (txt, vec) in enumerate(zip(chunk_texts, chunk_vecs)):
            self.chunks[i] = ChunkNode(chunk_id=i, text=txt, raw_embedding=vec)
        
        # A2. 提取所有句子和实体的原始信息 (批量异步)
        print(f"  [A2] Extracting Sentences and Entities...")
        batch_size = 20
        chunk_ids = list(self.chunks.keys())
        for i in range(0, len(chunk_ids), batch_size):
            batch_cids = chunk_ids[i:i+batch_size]
            tasks = [self._extract_raw_data(self.chunks[cid]) for cid in batch_cids]
            await asyncio.gather(*tasks)
            print(f"    -> Batch {i//batch_size + 1}/{(len(chunk_ids)+batch_size-1)//batch_size} done")
        
        print(f"  [A2] Total: {len(self.sentences)} sentences extracted")

        # === Phase B: Topic Discovery (L0) ===
        print(f"\n[Phase B] Fuzzy Topic Discovery...")
        await self._perform_fuzzy_topic_discovery(overlap_threshold)

        # === Phase C: Attention Cascading Update (L1 → L2 → L3) ===
        print(f"\n[Phase C] Executing Attention Cascading Update...")
        
        # C1. Update Chunk (L1): 【Attention 改造】动态融合 Topic + Sentences_Agg + Raw
        print(f"  [C1] Updating Chunks (L1) via attention_fusion...")
        for cid, chunk in self.chunks.items():
            child_sents = [self.sentences[sid] for sid in chunk.sentence_ids]
            if child_sents:
                s_vecs = np.stack([s.raw_embedding for s in child_sents])
                chunk.sent_agg_embedding = normalize(np.mean(s_vecs, axis=0))
            else:
                chunk.sent_agg_embedding = chunk.raw_embedding
            
            weighted_topic = np.zeros_like(chunk.raw_embedding)
            total_w = 0.0
            for tid, w in chunk.topic_memberships.items():
                weighted_topic += w * self.topics[tid].topic_vector
                total_w += w
            v_topic = normalize(weighted_topic / (total_w + 1e-9))

            # 【Attention 改造】Chunk 动态融合
            keys = [v_topic, chunk.sent_agg_embedding, chunk.raw_embedding]
            chunk.final_embedding = attention_fusion(chunk.raw_embedding, keys, temperature=0.1)
        
        # C2. Update Sentence (L2): 【Attention 改造】动态融合 Chunk_New + Entities_Agg + Raw
        print(f"  [C2] Updating Sentences (L2) via attention_fusion...")
        for sid, sent in self.sentences.items():
            if sent.entities:
                e_vecs = np.stack([e.raw_embedding for e in sent.entities])
                w_vecs = np.array([e.weight for e in sent.entities]).reshape(-1, 1)
                sent.ent_agg_embedding = normalize(np.sum(e_vecs * w_vecs, axis=0) / (np.sum(w_vecs) + 1e-9))
            else:
                sent.ent_agg_embedding = sent.raw_embedding

            if sent.chunk_ids:
                chunk_vecs = [self.chunks[cid].final_embedding for cid in sent.chunk_ids if cid in self.chunks]
                if chunk_vecs:
                    parent_chunk_embedding = normalize(np.mean(np.stack(chunk_vecs), axis=0))
                else:
                    parent_chunk_embedding = sent.raw_embedding
            else:
                parent_chunk_embedding = sent.raw_embedding

            # 【Attention 改造】Sentence 动态融合
            keys = [parent_chunk_embedding, sent.ent_agg_embedding, sent.raw_embedding]
            sent.final_embedding = attention_fusion(sent.raw_embedding, keys, temperature=0.1)

        # C3. Update Entity (L3): 【Attention 改造】动态融合 Sentence_New + Raw
        print(f"  [C3] Updating Entities (L3) via attention_fusion...")
        for sid, sent in self.sentences.items():
            for ent in sent.entities:
                # 【Attention 改造】Entity 动态融合
                keys = [sent.final_embedding, ent.raw_embedding]
                ent.final_embedding = attention_fusion(ent.raw_embedding, keys, temperature=0.1)
                
                if ent.name not in self.global_registry:
                    self.global_registry[ent.name] = []
                self.global_registry[ent.name].append(sent.sent_id)

        print(f"\n[Index Ready] Stats: {len(self.topics)} Topics, {len(self.chunks)} Chunks, {len(self.sentences)} Sentences, {len(self.global_registry)} Entities")

    async def _extract_raw_data(self, chunk: ChunkNode):
        """提取句子和实体的原始数据 (不做级联更新)，支持句子去重和多 chunk 关联"""
        s_texts = await extract_sentences(chunk.text)
        
        for s_text in s_texts:
            if len(s_text) < 5:
                continue
            
            existing_sid = self._find_existing_sentence(s_text)
            
            if existing_sid:
                existing_sent = self.sentences[existing_sid]
                if chunk.chunk_id not in existing_sent.chunk_ids:
                    existing_sent.chunk_ids.append(chunk.chunk_id)
                if existing_sid not in chunk.sentence_ids:
                    chunk.sentence_ids.append(existing_sid)
            else:
                s_vec = (await get_embedding_api([s_text]))[0]
                entities = await extract_entities_multi_round(s_text, max_rounds=2)
                if entities:
                    e_names = [e.name for e in entities]
                    e_vecs = await get_embedding_api(e_names)
                    for i, e in enumerate(entities):
                        e.raw_embedding = e_vecs[i]
                
                s_id = uuid.uuid4().hex[:8]
                sent_node = SentenceNode(
                    sent_id=s_id,
                    chunk_ids=[chunk.chunk_id],
                    text=s_text,
                    entities=entities,
                    raw_embedding=s_vec
                )
                
                self.sentences[s_id] = sent_node
                chunk.sentence_ids.append(s_id)
    
    def _find_existing_sentence(self, text: str) -> Optional[str]:
        """查找是否已存在相同文本的句子，返回 sent_id 或 None"""
        text_normalized = text.strip()
        for sid, sent in self.sentences.items():
            if sent.text.strip() == text_normalized:
                return sid
        return None

    async def _perform_fuzzy_topic_discovery(self, overlap_threshold=0.6):
        """模糊主题发现"""
        cids = list(self.chunks.keys())
        X = np.stack([self.chunks[i].raw_embedding for i in cids])
        N = len(cids)
        
        sim_matrix = np.dot(X, X.T)
        triu_sim = sim_matrix[np.triu_indices(N, k=1)]
        threshold = np.mean(triu_sim) + 0.5 * np.std(triu_sim) if len(triu_sim) > 0 else 0.5
        
        edges = []
        weights = []
        if LEIDEN_AVAILABLE:
            for i in range(N):
                for j in range(i+1, N):
                    if sim_matrix[i, j] > threshold:
                        edges.append((i, j))
                        weights.append(float(sim_matrix[i, j]))
            
            if edges:
                G = ig.Graph(n=N, edges=edges, directed=False)
                partition = leidenalg.find_partition(
                    G, leidenalg.RBConfigurationVertexPartition,
                    weights=weights, resolution_parameter=1.0
                )
                labels = partition.membership
            else:
                labels = list(range(N))
        else:
            labels = list(range(N))

        unique_labels = set(labels)
        for lbl in unique_labels:
            indices = [i for i, l in enumerate(labels) if l == lbl]
            centroid = np.mean(X[indices], axis=0)
            self.topics[lbl] = TopicNode(topic_id=lbl, topic_vector=normalize(centroid))
        
        Y = np.stack([self.topics[lbl].topic_vector for lbl in sorted(unique_labels)])
        
        count_fuzzy = 0
        for i, cid in enumerate(cids):
            primary = labels[i]
            self.chunks[cid].topic_memberships[primary] = 1.0
            
            sims = np.dot(X[i], Y.T)
            for tid_idx, score in enumerate(sims):
                real_tid = sorted(unique_labels)[tid_idx]
                if real_tid != primary and score > overlap_threshold:
                    self.chunks[cid].topic_memberships[real_tid] = float(score)
            
            if len(self.chunks[cid].topic_memberships) > 1:
                count_fuzzy += 1
        
        print(f"  -> {len(unique_labels)} topics discovered, {count_fuzzy} chunks have fuzzy memberships")

    # =========================================================================
    # Phase 2: Retrieval (Symmetric Cascading Process with Attention)
    # =========================================================================

    async def search(self, query: str, top_k_chunks=5, top_k_sents=5, enable_multi_hop=True,
                     relevance_threshold=0.75):
        """
        检索 - 对称级联流程 (使用 attention_fusion + attention_scoring)
        """
        print(f"\n{'='*60}")
        print(f"[Search] Query: {query}")
        print(f"{'='*60}")

        # === Step 1: Query Expansion & Raw Prep ===
        print(f"\n[Step 1] Query Expansion...")
        q_expanded = await get_completion_api(QUERY_EXPAND_PROMPT.format(query=query))
        print(f"  Expansion: {q_expanded[:100]}...")
        
        q_raw_vec = (await get_embedding_api([q_expanded]))[0]
        
        q_ent_objs = await extract_entities_multi_round(q_expanded, max_rounds=2)
        q_ent_raw_vecs = []
        if q_ent_objs:
            q_ent_raw_vecs = await get_embedding_api([e.name for e in q_ent_objs])
            for i, e in enumerate(q_ent_objs):
                e.raw_embedding = q_ent_raw_vecs[i]
            w_vecs = np.array([e.weight for e in q_ent_objs]).reshape(-1, 1)
            q_ent_agg = normalize(np.sum(np.stack(q_ent_raw_vecs) * w_vecs, axis=0) / (np.sum(w_vecs) + 1e-9))
        else:
            q_ent_agg = q_raw_vec
        
        print(f"  Entities: {[e.name for e in q_ent_objs]}")

        # === Step 2: Topic Routing (自适应 - 来自 v7.3) ===
        print(f"\n[Step 2] Topic Routing...")
        topic_scores = []
        for tid, node in self.topics.items():
            sim = np.dot(q_raw_vec, node.topic_vector)
            topic_scores.append((tid, sim))
        topic_scores.sort(key=lambda x: x[1], reverse=True)
        
        if not topic_scores:
            selected_topics = []
        else:
            max_score = topic_scores[0][1]
            relative_threshold = 0.75
            absolute_floor = 0.3
            selected_topics = []
            for tid, score in topic_scores:
                if score >= max_score * relative_threshold and score > absolute_floor:
                    selected_topics.append((tid, score))
                else:
                    break
        
        selected_tids = {t[0] for t in selected_topics}
        print(f"  Selected Topics: {[t[0] for t in selected_topics]} (Scores: {[round(t[1],3) for t in selected_topics]})")
        
        q_topic_vec = np.zeros_like(q_raw_vec)
        total_w = 0.0
        for tid, score in selected_topics:
            q_topic_vec += score * self.topics[tid].topic_vector
            total_w += score
        q_topic_vec = normalize(q_topic_vec / (total_w + 1e-9)) if total_w > 0 else q_raw_vec

        # === Step 3: Query Chunk Update (Level 1) - 【Attention 改造】===
        print(f"\n[Step 3] Query Chunk Update (attention_fusion)...")
        q_chunk_new = attention_fusion(q_raw_vec, [q_topic_vec, q_raw_vec], temperature=0.1)

        # === Step 4: Coarse Retrieval (Chunk Level) ===
        print(f"\n[Step 4] Coarse Retrieval (Chunk Level)...")
        chunk_candidates = []
        for cid, chunk in self.chunks.items():
            chunk_topics = set(chunk.topic_memberships.keys())
            if not chunk_topics.isdisjoint(selected_tids) or not selected_tids:
                score = np.dot(q_chunk_new, chunk.final_embedding)
                chunk_candidates.append((cid, score))
        
        chunk_candidates.sort(key=lambda x: x[1], reverse=True)
        top_chunks = chunk_candidates[:top_k_chunks]
        top_chunk_ids = {c[0] for c in top_chunks}
        top_chunk_scores = {c[0]: c[1] for c in top_chunks}
        print(f"  Locked Scope: {len(top_chunk_ids)} chunks (IDs: {list(top_chunk_ids)[:5]}...)")

        # === Step 5: Query Sentence Update (Level 2) - 【Attention 改造】===
        print(f"\n[Step 5] Query Sentence Update (attention_fusion)...")
        q_sent_new = attention_fusion(q_raw_vec, [q_chunk_new, q_ent_agg, q_raw_vec], temperature=0.1)

        # === Step 6: Query Entity Update (Level 3) - 【Attention 改造】===
        print(f"\n[Step 6] Query Entity Update (attention_fusion)...")
        q_ents_final = []
        if q_ent_objs:
            for ent in q_ent_objs:
                ent.final_embedding = attention_fusion(ent.raw_embedding, [q_sent_new, ent.raw_embedding], temperature=0.1)
                q_ents_final.append(ent.final_embedding)

        # === Step 7: Fine Retrieval & Scoring - 【Attention 改造】===
        print(f"\n[Step 7] Fine Retrieval & Attention Scoring...")
        results = []
        
        for cid in top_chunk_ids:
            chunk = self.chunks[cid]
            
            for sid in chunk.sentence_ids:
                sent = self.sentences[sid]
                
                # Entity 得分: MaxSim
                score_ent = 0.0
                if sent.entities and q_ents_final:
                    doc_E = np.stack([e.final_embedding for e in sent.entities])
                    query_E = np.stack(q_ents_final)
                    sims = np.dot(query_E, doc_E.T)
                    score_ent = np.mean(np.max(sims, axis=1))
                
                # 【Attention 改造】核心动态打分算子替代固定公式
                final_score, weights_dict = attention_scoring(
                    q_vec=q_raw_vec, 
                    doc_c=chunk.final_embedding, 
                    doc_s=sent.final_embedding, 
                    score_ent=score_ent, 
                    temperature=0.08
                )
                
                results.append({
                    "sent_id": sid,
                    "sentence": sent,
                    "text": sent.text,
                    "score": final_score,
                    "score_ent": score_ent,
                    "attn_weights": weights_dict,
                    "chunk_id": cid,
                    "hop": 1
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        
        # 相关性阈值检测
        if results:
            max_score = results[0]["score"]
            if max_score < relevance_threshold:
                print(f"  [Warning] 最高分 {max_score:.4f} 低于阈值 {relevance_threshold}，可能无相关内容")
                for r in results:
                    r["low_confidence"] = True
            else:
                for r in results:
                    r["low_confidence"] = False
        
        hop1_results = results[:top_k_sents]
        print(f"  Hop1 Results: {len(hop1_results)}")

        # === Step 8: Multi-Hop Expansion ===
        hop2_results = []
        if enable_multi_hop:
            print(f"\n[Step 8] Multi-Hop Expansion...")
            visited_sids = set(r["sent_id"] for r in hop1_results)
            hop_decay = 0.7
            
            for res in hop1_results:
                sent = res["sentence"]
                for ent in sent.entities:
                    neighbors = self.global_registry.get(ent.name, [])
                    for nid in neighbors:
                        if nid in visited_sids or nid not in self.sentences:
                            continue
                        
                        n_sent = self.sentences[nid]
                        primary_chunk_id = n_sent.chunk_ids[0] if n_sent.chunk_ids else None
                        if primary_chunk_id is None or primary_chunk_id not in self.chunks:
                            continue
                        n_chunk = self.chunks[primary_chunk_id]
                        
                        # Entity 得分
                        s_ent = 0.0
                        if n_sent.entities and q_ents_final:
                            doc_E = np.stack([e.final_embedding for e in n_sent.entities])
                            query_E = np.stack(q_ents_final)
                            sims = np.dot(query_E, doc_E.T)
                            s_ent = np.mean(np.max(sims, axis=1))
                        
                        # 【Attention 改造】同样用于 Hop 2
                        base_score, w_dict = attention_scoring(q_raw_vec, n_chunk.final_embedding, n_sent.final_embedding, s_ent, 0.08)
                        final_score = base_score * hop_decay
                        
                        # 跨主题标记
                        n_topics = set(n_chunk.topic_memberships.keys())
                        is_cross = n_topics.isdisjoint(selected_tids)
                        
                        hop2_results.append({
                            "sent_id": nid,
                            "sentence": n_sent,
                            "text": n_sent.text,
                            "score": final_score,
                            "attn_weights": w_dict,
                            "chunk_ids": n_sent.chunk_ids,
                            "hop": 2,
                            "via": ent.name,
                            "is_cross_topic": is_cross
                        })
                        visited_sids.add(nid)
            
            hop2_results.sort(key=lambda x: x["score"], reverse=True)
            hop2_results = hop2_results[:top_k_sents]
            print(f"  Hop2 Results: {len(hop2_results)}")

        is_low_confidence = hop1_results and hop1_results[0].get("low_confidence", False)
        
        top_chunks_info = []
        for cid, score in top_chunks:
            chunk = self.chunks[cid]
            top_chunks_info.append({
                "chunk_id": cid,
                "text": chunk.text,
                "score": score
            })
        
        return {
            "hop1": hop1_results,
            "hop2": hop2_results,
            "top_chunks": top_chunks_info,
            "low_confidence": is_low_confidence,
            "max_score": hop1_results[0]["score"] if hop1_results else 0.0
        }

    # =========================================================================
    # Phase 3: Generation
    # =========================================================================

    def _expand_context(self, sent: SentenceNode, window_size: int = 1) -> str:
        """扩展句子上下文"""
        if not sent.chunk_ids or sent.chunk_ids[0] not in self.chunks:
            return sent.text
            
        chunk_node = self.chunks[sent.chunk_ids[0]]
        full_text = chunk_node.text
        target_sentence = sent.text
        
        try:
            sentences = re.split(r'(?<=[。！？；])', full_text)
            sentences = [s.strip() for s in sentences if s.strip()]
            
            target_idx = -1
            for i, s in enumerate(sentences):
                if target_sentence in s or s in target_sentence:
                    target_idx = i
                    break
            
            if target_idx == -1:
                return target_sentence
            
            start = max(0, target_idx - window_size)
            end = min(len(sentences), target_idx + window_size + 1)
            
            return "".join(sentences[start:end])
            
        except Exception as e:
            print(f"[Context Warning] Expansion failed: {e}")
            return target_sentence

# ============================================================================
# 7.5 答案生成与校验 (来自 v7.3 详细版)
# ============================================================================

async def _generate_answer_impl(engine, query: str, search_results: dict):
    """生成最终答案 (引入双轨对比验证策略)"""
    print(f"\n[Generation] Executing Dual-Track Contrastive Synthesis...")
    
    max_score = search_results.get("max_score", 0)
    
    # 【双阈值控制系统（保留）】
    # HARD_THRESHOLD = 0.55  # 硬性熔断阈值 (完全无关的极低分)
    # SOFT_THRESHOLD = 0.75  # 软提示阈值 (可能是碎片或泛化概念，需要 LLM 仔细甄别)
    HARD_THRESHOLD = 0.30  # 硬性熔断阈值 (完全无关的极低分)
    SOFT_THRESHOLD = 0.50  # 软提示阈值 (可能是碎片或泛化概念，需要 LLM 仔细甄别)
    # 判断 Verifier 是否把结果全删光了
    all_wiped = not search_results.get("hop1") and not search_results.get("hop2")
    
    # 1. 硬性熔断：极低分 或 Verifier 把结果全删光了
    if all_wiped or max_score < HARD_THRESHOLD:
        print(f"  [Warning] 极低置信度或无保留结果 (max_score={max_score:.4f}, all_wiped={all_wiped})，直接安全熔断。")
        return f"⚠️ **未找到相关信息**\n\n抱歉，在当前知识库中未找到与「{query}」相关的内容。\n\n建议尝试使用不同的关键词重新提问。"
    
    # 2. 软提示注入（保留）
    confidence_warning = ""
    if max_score < SOFT_THRESHOLD:
        print(f"  [Warning] 检索最高分 {max_score:.4f} 偏低，向生成模型注入【软提示】。")
        confidence_warning = f"\n\n【系统预警】：本次检索到的证据片段匹配度较低（最高分 {max_score:.4f}）。请务必极其谨慎地拼凑线索。如果发现提供的片段虽然沾边，但实际上无法有效、完整地解答用户问题，请直言\"资料不足\"，切勿强行附会或编造。"

    # ============================================================
    # 双轨组装：物理隔离传统 RAG 结果与 HyperRAG 图检索结果
    # ============================================================
    
    # 第一路：组装标准 RAG 结果 (作为基础事实准线)
    context_standard_rag = []
    for i, chunk_info in enumerate(search_results.get("top_chunks", [])[:5]):  # 只取 Top 3 最相关的 Chunk
        context_standard_rag.append(f"【传统检索段落 {i+1}】(匹配度 {chunk_info['score']:.2f}):\n{chunk_info['text']}")
    
    # 第二路：组装 HyperRAG 结果 (作为深度关联与推理线索)
    context_hyperrag = []
    for i, res in enumerate(search_results.get("hop1", [])):
        text = res.get("expanded_text", res["text"])
        context_hyperrag.append(f"【图检索直接证据 {i+1}】: {text}")
        
    for i, res in enumerate(search_results.get("hop2", [])):
        text = res.get("expanded_text", res["text"])
        cross_topic = " (跨主题延伸)" if res.get('is_cross_topic') else ""
        context_hyperrag.append(f"【图检索二跳逻辑 {i+1}】{cross_topic} (线索实体 '{res.get('via','')}'): {text}")

    # ============================================================
    # 双轨对比生成 Prompt
    # ============================================================
    system_prompt = f"""你是一个严谨的知识校验专家与信息整合助手。
你收到了针对同一个用户问题的两路检索结果：【传统检索（Standard RAG）】和【图谱发散检索（HyperRAG）】。

你的核心任务是：交叉验证这两路信息，去伪存真，生成最准确的答案。

【严格执行以下生成策略】：
1. **确立基准（锚定事实）**：首先阅读【传统检索】的内容。这部分通常是字面最匹配的基础事实。如果它已经直接、完整地回答了问题，请以它为主干构建答案。
2. **交叉验证（防张冠李戴）**：审视【图谱发散检索】的内容。图检索容易过度发散。如果图检索中的实体（如 GraphRAG）与用户询问的实体（如 LightRAG）不一致，**必须坚决丢弃该图检索线索**，严禁将两者的概念强行缝合。
3. **补充深度（按需使用）**：只有当【图谱发散检索】确实补充了目标实体的深层逻辑、多跳关系或背景原因时，才将其作为补充说明融入答案。
4. **简洁清晰**：直接回答问题，不要向用户解释你的对比过程，不要使用诸如"根据传统检索..."或"图检索表明..."的句式。如果发现资料存在冲突，以【传统检索】中的原始文本逻辑为准。
5. **拒绝复读**：保持语言凝练，严禁在输出末尾死循环重复生成相同的列表或段落。{confidence_warning}
"""

    user_content = f"""
用户问题：{query}

=========================================
第一路：【传统检索（提供基础事实准线）】
{chr(10).join(context_standard_rag) if context_standard_rag else "无传统检索结果"}
=========================================

第二路：【图谱发散检索（提供深度关联线索）】
{chr(10).join(context_hyperrag) if context_hyperrag else "无图谱检索结果"}
=========================================

请执行交叉验证策略，生成最终回答：
"""
    
    full_prompt = system_prompt + user_content
    answer = await get_completion_api(full_prompt)
    return answer

HyperRAG_Attention_v81.generate_answer = lambda self, query, search_results: _generate_answer_impl(self, query, search_results)

async def verify_results(engine, query: str, results: dict) -> dict:
    """对检索结果进行实体一致性校验"""
    print("\n" + "="*50)
    print("[实体一致性校验] 开始基于'滑窗扩展上下文'过滤张冠李戴的结果...")
    print("="*50)
    
    # 【核心改进：滑窗扩展前置】提前把扩展好的上下文塞进字典，供 Verifier 阅读
    for r in results.get("hop1", []):
        r["expanded_text"] = engine._expand_context(r["sentence"], window_size=2)
    for r in results.get("hop2", []):
        r["expanded_text"] = engine._expand_context(r["sentence"], window_size=2)
    
    if results.get("hop1"):
        results["hop1"] = await ConsistencyVerifier.verify_batch(query, results["hop1"])
    
    if results.get("hop2"):
        results["hop2"] = await ConsistencyVerifier.verify_batch(query, results["hop2"])
    
    if not results.get("hop1") and not results.get("hop2"):
        print("⚠️ 经过校验，所有检索结果均被认定为'张冠李戴'或无关。")
        results["low_confidence"] = True
    
    return results

HyperRAG_Attention_v81.verify_results = lambda self, query, results: verify_results(self, query, results)

# ============================================================================
# 8. Main Execution
# ============================================================================

async def main():
    print("=" * 60)
    print("HyperRAG v8.1: Attention Cascading + Full Features")
    print("=" * 60)
    
    # 加载 Markdown 文件
    md_file_path = "./honglong.md"
    
    print(f"\n[文档] 加载文件: {md_file_path}")
    raw_text = load_markdown_file(md_file_path)
    cleaned_text = clean_markdown_text(raw_text)
    print(f"[文档] 原始长度: {len(raw_text)} 字符, 清理后: {len(cleaned_text)} 字符")
    
    # 创建引擎
    engine = HyperRAG_Attention_v81()
    
    # 检查缓存
    cache_path = get_cache_path(cleaned_text, prefix="hyperrag_v81")
    
    if load_engine_state(engine, cache_path):
        print("[缓存] 使用缓存数据，跳过索引构建")
    else:
        print("\n[缓存] 未找到缓存，开始构建索引...")
        
        await engine.build_index(
            cleaned_text,
            max_token_size=400,
            overlap_token_size=50,
            overlap_threshold=0.6
        )
        
        save_engine_state(engine, cache_path)
    
    print("\n" + "=" * 60)
    print("执行检索验证")
    print("=" * 60)
    
    # 测试查询
    queries = [
        "总结一下整体内容"
    ]
    
    for q in queries:
        results = await engine.search(q, top_k_chunks=10, top_k_sents=10)
        
        print(f"\n{'='*60}")
        print(f"Query: {q}")
        print(f"{'='*60}")
        
        # 显示一跳结果 (含注意力权重)
        print(f"\n{'─'*30} 一跳结果 (Top {len(results['hop1'])}) {'─'*30}")
        for i, res in enumerate(results["hop1"]):
            w = res.get("attn_weights", {})
            print(f"{i+1}. [{res['score']:.4f}] (w_c:{w.get('w_c',0):.2f} w_s:{w.get('w_s',0):.2f} w_e:{w.get('w_e',0):.2f})")
            print(f"   {res['text'][:80]}...")
        
        # 显示二跳结果
        print(f"\n{'─'*30} 二跳结果 (Top {len(results['hop2'])}) {'─'*30}")
        for i, res in enumerate(results["hop2"]):
            via = res.get('via', '未知')
            cross = "[跨主题]" if res.get('is_cross_topic') else "[同主题]"
            print(f"{i+1}. [{res['score']:.4f}] {cross} via '{via}'")
            print(f"   {res['text'][:80]}...")
        
        # 校验步骤
        results = await engine.verify_results(q, results)
        
        print(f"\n{'─'*30} 校验后结果 {'─'*30}")
        print(f"Hop1 保留: {len(results.get('hop1', []))} 条")
        print(f"Hop2 保留: {len(results.get('hop2', []))} 条")
        
        # 生成最终答案
        final_answer = await engine.generate_answer(q, results)
        
        print(f"\n{'='*60}")
        print("HyperRAG Final Answer:")
        print(f"{'='*60}")
        print(final_answer)

if __name__ == "__main__":
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(main())

