from multiprocessing import context
import re
from unittest import result

from matplotlib.dates import HOURS_PER_DAY
from neo4j.exceptions import ResultFailedError
from torch import chunk
from torch.nn.utils import weight_norm
from torch.cuda import temperature
import uuid
from dataclasses import dataclass,field
from typing import Optional
import os
import sys
from dotenv import load_dotenv
from openai import AsyncOpenAI
import numpy as np
import asyncio
import igraph as ig
import leidenalg 
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Hyper-RAG"))
from hyperrag.utils import encode_string_by_tiktoken, decode_tokens_by_tiktoken
load_dotenv()
api_key_embedding = os.getenv("api_key_embedding")
api_url_embedding = os.getenv("api_url_embedding")
api_model_embedding = os.getenv("api_model_embedding")
api_key_llm = os.getenv("api_key_llm")
api_url_llm = os.getenv("api_url_llm")
api_model_llm = os.getenv("api_model_llm")

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm > 0 else v
client_embedding=AsyncOpenAI(api_key=api_key_embedding,base_url=api_url_embedding)
client_llm=AsyncOpenAI(api_key=api_key_llm,base_url=api_url_llm)
async def get_embedding_api(text:list[str])->list[list[np.ndarray]]:
    if not text:
        return []
    
    #调用openai的嵌入接口
    embedding_api = await client_embedding.embeddings.create(
        model=api_model_embedding,
        input=text,
    )
    embedding_list=[c.embedding for c in embedding_api.data] 
    return embedding_list


async def get_llm_api(text:str,prompt:str)->str:
    if not text:
        return ""
    
    #调用openai的llm接口
    llm_api = await client_llm.chat.completions.create(
        model=api_model_llm,
        messages=[
            {"role": "user", "content": text},
        ],
    )
    return llm_api.choices[0].message.content
#data format
@dataclass
class EntityNode:
    entity_id:str
    entity_name:str
    entity_type:str
    entity_importance:float
    entity_description:str
    entity_raw_embedding:list[np.ndarray]
    entity_final_embedding:list[np.ndarray]=field(default_factory=list)
    entity_sentence_id:list[str]=field(default_factory=list)

@dataclass 
class SentenceNode:
    sentence_id:int
    sentence_embedding:list[np.ndarray]
    sentence_text:str
    sentence_final_embedding:list[np.ndarray]=field(default_factory=list)
    sentence_entity_id:list[str]=field(default_factory=list)
    chunk_id:list[str]=field(default_factory=list)

@dataclass
class ChunkNode:
    chunk_id:str
    chunk_raw_embedding:list[np.ndarray]
    chunk_text:str 
    chunk_sentence_id:list[str]=field(default_factory=list)
    chunk_topic_id:list[str]=field(default_factory=list)
    chunk_topic_weight:dict[str,float]=field(default_factory=dict)
    chunk_final_embedding:list[np.ndarray]=field(default_factory=list)
@dataclass
class TopicNode:
    # topic_id:int
    topic_id:str
    topic_embedding:list[np.ndarray]
    topic_chunk_id:list[str]=field(default_factory=list)
    # topic_description:str
    
    #提示词


SENTENCE_EXTRACT_PROMPT="""
请从以下段落中提取所有句子，这些句子可以是一句话，也可以是多句话，但必须保证这是一段完整的语义。
段落：{paragraph}
请以JSON格式输出：{{ "sentences": [ {{"text": "..."}}, ... ] }}
"""
ENTITY_EXTRACTION_PROMPT="""
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
QUERY_EXPAND_PROMPT = """
你是一个搜索查询优化器。
请将用户的问题扩展为一个更详细的问题，包含更多相关的关键词、上下文和可能涉及的实体。

重要规则：
1. 如果原查询是问句，扩展后必须保持问句形式
2. 不要回答问题，只是扩展问题本身
3. 保留原查询的核心疑问词（什么、如何、为什么等）

示例输入："李长风的最终结局是什么？"
示例输出："作为出使西域的使节，李长风在经历了各种艰险和外交博弈之后，他的最终命运和结局是什么？他是否完成了使命并安全返回？"

用户查询：{query}
请直接输出扩展后的问题，不要有任何解释：
"""
ANSWER_PROMPT="""
你是一个智能助手，根据用户的问题{query}和拿到的相关的信息包括主要的切块信息{chunk_info}，以及一跳信息{hop1_results}，和相应的二跳信息{hop2_results},
并且{chunk_info}是主要信息，你需要通过它来了解整个背景，{hop1_results}是一跳信息，你需要通过它来了解更详细的背景，{hop2_results}是二跳信息，你需要仔细斟酌，来确定最终的答案
请以JSON格式输出，只输出一个JSON对象，不要有任何注释或额外文字：
{{"answer": "回答内容"}}
"""
#对拿到的文本切块
def chunking_by_token_size(
    content: str, overlap_token_size=128, max_token_size=1024, tiktoken_model="gpt-4o"
):
    tokens = encode_string_by_tiktoken(content, model_name=tiktoken_model)
    results = []
    for index, start in enumerate(
        range(0, len(tokens), max_token_size - overlap_token_size)
    ):
        chunk_content = decode_tokens_by_tiktoken(
            tokens[start : start + max_token_size], model_name=tiktoken_model
        )
        results.append(
            {
                "tokens": min(max_token_size, len(tokens) - start),
                "content": chunk_content.strip(),
                "chunk_order_index": index,
            }
        )
    return results
async def chunk_text(text:str)->list[str]:
    if not text:
        return []
    
    chunks=chunking_by_token_size(
        text,
        overlap_token_size=128,
        max_token_size=1024,
        tiktoken_model="gpt-4o"
    )
    result=[]
    for chunk in chunks:
        result.append({"context":chunk["content"],"chunk_order_index":chunk["chunk_order_index"]})
    return result

async def extract_sentences(text:str)->list[str]:
    if not text:
        return []
    sentences=await get_llm_api(SENTENCE_EXTRACT_PROMPT.format(paragraph=text))
    data=sentences.get("sentences",[])
    result=[]
    for sentence in data:
        result.append(sentence.get("text",""))
    return result
async def extract_entities(text:str)->list[dict]:
    if not text:
        return []
    entities=await get_llm_api(ENTITY_EXTRACTION_PROMPT.format(sentence=text))
    data=entities.get("entities",[])
    return data
def attention_fusion(query_vec:np.ndarray,key_vecs:list[np.ndarray],temperature:float=0.1)->np.ndarray:
    if not key_vecs:
        return query_vec
    sims=np.array([np.dot(query_vec,k) for k in key_vecs])
    logits=sims/temperature
    exp_logits=np.exp(logits-np.max(logits))
    weights=exp_logits/(np.sum(exp_logits)+1e-9)
    fused_vec=np.zeros_like(query_vec)
    for w,k in zip(weights,key_vecs):
        fused_vec+=w*k
    return normalize(fused_vec)
def attention_score(sim_c:float,sim_s:float,sim_e:float,temperature:float=0.1)->float:

    sims=np.array([sim_c,sim_s,sim_e])
    logits=sims/temperature
    exp_logits=np.exp(logits-np.max(logits))
    weights=exp_logits/(np.sum(exp_logits)+1e-9)
    final_score=np.dot(weights,[sim_c,sim_s,sim_e])
    return final_score

class HyperGraph:
    def __init__(self):
        self.chunks:dict[str,ChunkNode]={} #chunk_id->ChunkNode
        self.sentences:dict[str,SentenceNode]={} #sent_id->SentenceNode
        self.topics:dict[str,TopicNode]={} #topic_id->TopicNode
        self.entities:dict[str,EntityNode]={} #entity_id->EntityNode

    async def build_index(self,raw_text:str,overlap_threshold:float):
            #切块
            if not raw_text:
                return[]
            chunk_data=await chunk_text(raw_text)
            chunk =[c["context"] for c in chunk_data]
            chunk_raw_embedding=await get_embedding_api(chunk)
            for (chunk_data,embedding) in zip(chunk,chunk_raw_embedding):
                chunk_id=str(uuid.uuid4())
                self.chunks[chunk_id]=ChunkNode(
                    chunk_id=chunk_id,
                    chunk_raw_embedding=embedding,
                    chunk_text=chunk_data
                )
            batch_size=5
            for i in range(0,len(self.chunks),batch_size):
                batch_chunk_ids=list(self.chunks.keys())[i:i+batch_size]
                batch_chunks=[self.chunks[chunk_id] for chunk_id in batch_chunk_ids]
                tasks=[self._extract_sentences_and_entities(chunk) for chunk in batch_chunks]
                await asyncio.gather(*tasks)
                print(f"进度为{i//batch_size+1}/{(len(self.chunks)+batch_size-1)//batch_size}")
            
            #生成主题
            await self._generate_topics(overlap_threshold)

            for cid,chunk in self.chunks.items():
                child_sents = [self.sentences[sid] for sid in chunk.chunk_sentence_id]
                if child_sents:
                    sentence_agg=np.stack([sent.sentence_embedding for sent in child_sents])
                    sentence_chunk_agg=normalize(np.mean(sentence_agg,axis=0))
                child_topic=[self.topics[tid] for tid in chunk.chunk_topic_id]
                weight_norm=[chunk.chunk_topic_weight[topic_id] for topic_id in chunk.chunk_topic_id]
                if child_topic:
                    topic_agg=np.stack([topic.topic_embedding for topic in child_topic])
                    topic_chunk_agg=normalize(np.dot(weight_norm,topic_agg)/sum(weight_norm))
                keys=[topic_chunk_agg,sentence_chunk_agg,chunk.chunk_raw_embedding]
                chunk_final_embedding=attention_fusion(chunk.chunk_raw_embedding,keys,temperature=0.1) 
                chunk.chunk_final_embedding=chunk_final_embedding
            
            for sid,sentence in self.sentences.items():
                child_chunk=[self.chunks[cid] for cid in sentence.chunk_id]
                if child_chunk:
                    chunk_agg=np.stack([chunk.chunk_final_embedding for chunk in child_chunk])
                    chunk_sentence_agg=normalize(np.mean(chunk_agg,axis=0))
                child_entity=[self.entities[eid] for eid in sentence.sentence_entity_id]
                weight_norm=[self.entities[eid].entity_importance for eid in sentence.sentence_entity_id]
                if child_entity:
                    entity_agg=np.stack([entity.entity_raw_embedding for entity in child_entity])
                    entity_sentence_agg=normalize(np.dot(weight_norm,entity_agg)/sum(weight_norm))
                keys=[chunk_sentence_agg,entity_sentence_agg,sentence.sentence_raw_embedding]
                sentence_final_embedding=attention_fusion(sentence.sentence_raw_embedding,keys,temperature=0.1) 
                sentence.sentence_final_embedding=sentence_final_embedding
            
            for eid,entity in self.entities.items():
                child_sentence=[self.sentences[sid] for sid in entity.entity_sentence_id]
                if child_sentence:
                    sentence_agg=np.stack([sentence.sentence_final_embedding for sentence in child_sentence])
                    sentence_entity_agg=normalize(np.mean(sentence_agg,axis=0))
                keys=[sentence_entity_agg,entity.entity_raw_embedding]
                entity_final_embedding=attention_fusion(entity.entity_raw_embedding,keys,temperature=0.1) 
                entity.entity_final_embedding=entity_final_embedding
            print(f"build index finished")
    
    async def _extract_sentences_and_entities(self,chunkNode:ChunkNode):
        if not chunkNode.chunk_text:
            return []
        clear_sentences=await extract_sentences(chunkNode.chunk_text)
        sentence_raw_embedding=await get_embedding_api(clear_sentences)
        for (sentence,embedding_sentence) in zip(clear_sentences,sentence_raw_embedding):
            exist_sentence_id=self._find_existing_sentence(sentence)
            if exist_sentence_id:
                self.sentences[exist_sentence_id].chunk_id.append(chunkNode.chunk_id)
                chunkNode.chunk_sentence_id.append(exist_sentence_id)
                continue
            sentence_id=str(uuid.uuid4())
            entity_data=await extract_entities(sentence)
            entity_raw_embedding=await get_embedding_api([entity["name"] for entity in entity_data])
            for (entity,embedding) in zip(entity_data,entity_raw_embedding):
                entity_id=str(uuid.uuid4())
                self.entities[entity_id]=EntityNode(
                    entity_id=entity_id,
                    entity_name=entity["name"],
                    entity_type=entity["type"],
                    entity_importance=entity["importance"],
                    entity_description=entity["description"],
                    entity_raw_embedding=embedding,
                    entity_sentence_id=[sentence_id]
                )
                self.sentences[sentence_id].sentence_entity_id.append(entity_id)
            self.sentences[sentence_id]=SentenceNode(
                sentence_id=sentence_id,
                sentence_embedding=embedding_sentence,
                sentence_text=sentence,
                chunk_id=[chunkNode.chunk_id]
            )
            chunkNode.chunk_sentence_id.append(sentence_id)
    
    
    
    
    async def _find_existing_sentence(self,sentence:str)->Optional[str]:
        text_normalized =sentence.strip()
        for sentenceNode in self.sentences.values():
            if sentenceNode.sentence_text.strip()==text_normalized:
                return sentenceNode.sentence_id
        return None
    
    
    
    
    async def _generate_topics(self,overlap_threshold:float):
        chunk=list(self.chunks.keys())
        N=len(chunk)
        X=np.stack([self.chunks[chunk_id].chunk_raw_embedding for chunk_id in chunk])
        sim_matrix=np.dot(X,X.T)
        tri_matrix=sim_matrix[np.triu_indices(N,k=1)]
        threshold = np.mean(tri_matrix)+0.5*np.std(tri_matrix) if np.std(tri_matrix)>0 else np.mean(tri_matrix)
        edges=[]
        weight=[]
        for i in range(N):
            for j in range(i+1,N):
                if sim_matrix[i,j]>=threshold:
                    edges.append((i,j))    
                    weight.append(sim_matrix[i,j])
        if edges:
            G=ig.Graph(n=N,edges=edges,directed=False)
            partition=leidenalg.find_partition(G,
                                            leidenalg.ModularityVertexPartition,
                                            weights=weight,resolution_parameter=1.0)
            
            label=partition.membership  
        else:
            label=list(range(N))
        set_label=set(label)
        topic_leiden_embedding_list=[]
        for lbl in set_label:
            list_chunk=[label_index for label_index,label_value in enumerate(label) if label_value==lbl]
            topic_leiden_embedding=np.mean(X[list_chunk],axis=0) 
            topic_nor_leiden_embedding=normalize(topic_leiden_embedding)
            topic_leiden_embedding_list.append(topic_nor_leiden_embedding)

        Y=np.stack(topic_leiden_embedding_list)
        weighet= np.dot(X,Y.T)
        M=len(set_label)
        
        ######区别
        chunk_list=[]
        topic_nonro_embedding=np.zeros(len(X[0]))
        weight_sum=0
        create_topic_id=[]
        for j in range(M):
            topic_id=str(uuid.uuid4())
            create_topic_id.append(topic_id)
            for i in range(N):
                if weighet[i,j]>overlap_threshold:
                    topic_nonro_embedding=weighet[i,j]*X[i]+topic_nonro_embedding
                    weight_sum=weighet[i,j]+weight_sum
                    chunk_list.append(chunk[i])
            topic_embedding=normalize(topic_nonro_embedding/weight_sum)

            self.topics[topic_id]=TopicNode(
                topic_id=topic_id,
                topic_embedding=topic_embedding,
                topic_chunk_id=chunk_list
            )
            for chunk_id in chunk_list:
                self.chunks[chunk_id].chunk_topic_id.append(topic_id)
            chunk_list=[]
            topic_nonro_embedding=np.zeros(len(X[0]))
            weight_sum=0
        T=np.stack([self.topics[t_id].topic_embedding for t_id in create_topic_id])
        final_weights=np.dot(X,T.T)
        for i in range(N):
            for j in range(M):
                t_id = create_topic_id[j]
                # 只把被划入该主题的切块与之对应的计算权值保存入字典
                if t_id in self.chunks[chunk[i]].chunk_topic_id:
                    self.chunks[chunk[i]].chunk_topic_weight[t_id] = float(final_weights[i,j])
               
        print("主题生成完成")

    async def _extract_query_entities(self,query:str):
        sentence_id=str(uuid.uuid4())
        chunk_id=str(uuid.uuid4())
        entity_data=await extract_entities(query)
        query_expand_embedding=await get_embedding_api([query])
        query_expand_embedding=query_expand_embedding[0]
        
        entity_raw_embedding=await get_embedding_api([entity["name"] for entity in entity_data])
        entity_query={}
        sentence_query={}
        chunk_query={}
        for (entity,embedding) in zip(entity_data,entity_raw_embedding):
            entity_id=str(uuid.uuid4())
            self.entities[entity_id]=entity_query[entity_id]=EntityNode(
                    entity_id=entity_id,
                    entity_name=entity["name"],
                    entity_type=entity["type"],
                    entity_importance=entity["importance"],
                    entity_description=entity["description"],
                    entity_raw_embedding=embedding,
                    entity_sentence_id=[sentence_id]
                )
            
        self.sentences[sentence_id]=sentence_query[sentence_id]=SentenceNode(
                sentence_id=sentence_id,
                sentence_embedding=query_expand_embedding,
                sentence_text=query,
                chunk_id=[chunk_id],
                sentence_entity_id=list(entity_query.keys())
            )
        self.chunks[chunk_id]=chunk_query[chunk_id]=ChunkNode(
                    chunk_id=chunk_id,
                    chunk_sentence_id=[sentence_id],
                    chunk_raw_embedding=query_expand_embedding,
                    chunk_text=query
                )
        return entity_query,sentence_query,chunk_query
        
    async def search(self,query:str,top_k:int=5,top_sent_k:int=5,relative_threshold:float=0.75,absolute_threshold:float=0.3)->dict[str]:
        query_expand=await get_llm_api(QUERY_EXPAND_PROMPT.format(query=query))
        entity_query,sentence_query,chunk_query=await self._extract_query_entities(query_expand)
        query_expand_embedding=next(iter(chunk_query.values())).chunk_raw_embedding
        
        X=np.stack([query_expand_embedding])
        Y=np.stack([topic.topic_embedding for topic in self.topics.values()])
        Y_length=Y.shape[0]
        topic_weight=np.dot(X,Y.T)
        topic_score=[]
        for i in range(Y_length):
            topic_score.append((i, topic_weight[0, i]))
        topic_score.sort(key=lambda x: x[1], reverse=True)
        max_score=topic_score[0][1]
        query_topic_agg=0.0
        sum_weight=0.0
        topic_id_list=[]
        for i in range(Y_length):
            if topic_weight[0][i]>max_score*relative_threshold and topic_weight[0][i]>absolute_threshold:
                topic_id=self.topics.keys()[i]
                topic_node=self.topics[topic_id]
                query_topic_agg+=topic_weight[0][i]*topic_node.topic_embedding
                sum_weight+=topic_weight[0][i]
                topic_id_list.append(topic_id)
        query_topic_agg=normalize(query_topic_agg/sum_weight)

              
        entity_embedding_list=[entity.entity_raw_embedding for entity in iter(entity_query.values())]
        entities_score=[entity.entity_importance for entity in iter(entity_query.values())]
        np_entity_embedding_list=np.stack(entity_embedding_list)
        entity_weight=np.stack(entities_score)
        entities_agg=np.dot(entity_weight,np_entity_embedding_list.T)/sum(entity_weight)
        entities_agg=normalize(entities_agg)

        query_chunk_embedding=attention_fusion(query_expand_embedding,[query_topic_agg,query_expand_embedding],temperature=0.1)
        
        query_sentence_embedding=attention_fusion(query_expand_embedding,[entities_agg,query_chunk_embedding,],temperature=0.1)
        query_entity_final=[]
        for entity in entity_embedding_list:
            query_entity_embedding=attention_fusion(entity,[query_sentence_embedding,entity],temperature=0.1)
            query_entity_final.append(query_entity_embedding)
        score=[]
        for i in topic_id_list:
            topic_node=self.topics[self.topics.keys()[i]]
            for chunk_id in topic_node.topic_chunk_id:
                chunk_node=self.chunks[chunk_id]
                chunk_score=np.dot(query_chunk_embedding,chunk_node.chunk_final_embedding)
                score.append((chunk_id,chunk_score))
        score=list(set(score))
        score.sort(key=lambda x:x[1],reverse=True)
        results=[]
        for chunk_search in score[:top_k]:
            for sentence_id in self.chunks[chunk_search[0]].chunk_sentence_id:
                sentence_node=self.sentences[sentence_id]
                entity_search_node_list=[self.entities[entity_id] for entity_id in sentence_node.sentence_entity_id]
                score_entity=0.0
                if entity_search_node_list and query_entity_final:
                    doc_e=np.stack([entity.entity_final_embedding for entity in entity_search_node_list])
                    query_e=np.stack(query_entity_final)
                    sims=np.dot(query_e,doc_e.T)
                    score_entity=np.mean(np.max(sims,axis=1))

                sim_c=np.dot(query_chunk_embedding,self.chunks[chunk_search[0]].chunk_final_embedding)
                sim_s=np.dot(query_sentence_embedding,sentence_node.sentence_final_embedding)
                final_score=attention_score(sim_c,sim_s,score_entity,temperature=0.08)
                results.append({"score":final_score,"sentence":sentence_node,"entity":entity_search_node_list})
        results.sort(key=lambda x:x["score"],reverse=True)
        hop1_results=results[:top_sent_k]
        hop2_results=[]
        hop_decay = 0.7
        for hop1_result in hop1_results:
            for entity in hop1_result["entity"]:
                for sentence_id in entity.entity_sentence_id:
                    if sentence_id!=hop1_result["sentence"].sentence_id:
                        sentence_hop_node=self.sentences[sentence_id]
                        entity_search_hop_node_list=[self.entities[entity_id] for entity_id in sentence_hop_node.sentence_entity_id]
                        score_hop_entity=0.0
                        if entity_search_hop_node_list and query_entity_final:
                            doc_e=np.stack([entity.entity_final_embedding for entity in entity_search_hop_node_list])
                            query_e=np.stack(query_entity_final)
                            sims=np.dot(query_e,doc_e.T)
                            score_hop_entity=np.mean(np.max(sims,axis=1))
                        sim_hop_c=np.dot(query_chunk_embedding,self.chunks[sentence_hop_node.chunk_id[0]].chunk_final_embedding)
                        sim_hop_s=np.dot(query_sentence_embedding,sentence_hop_node.sentence_final_embedding)
                        final_hop_score=attention_score(sim_hop_c,sim_hop_s,score_hop_entity,temperature=0.08)*hop_decay
                        hop2_results.append({"score":final_hop_score,"sentence":sentence_hop_node,"entity":entity_search_hop_node_list})
        hop2_results.sort(key=lambda x:x["score"],reverse=True)
        hop2_results=hop2_results[:top_sent_k]
        top_chunk_info=score[:top_k]
        return {"hop1_results":hop1_results,"hop2_results":hop2_results,"top_chunk_info":top_chunk_info}

    def _expand_context(self,sent:SentenceNode,chunk:ChunkNode,windows_size:int=1)->str:
        if not sent.chunk_id:
            return sent.sentence_text
        chunk_nodes=[self.chunks[c] for c in sent.chunk_id]
        #不一样
        for chunk_node in chunk_nodes:
            if chunk_node.chunk_id==chunk.chunk_id:
                target=-1
                expand_sentences=[]
                for i,sid in enumerate(chunk_node.chunk_sentence_id):
                    if sid==sent.sentence_id:
                        target=i
                        break
                if target==-1:
                    return sent.sentence_text   
                for i in range(max(0,target-windows_size),min(len(chunk_node.chunk_sentence_id),target+windows_size+1)):
                    expand_sentences.append(self.sentences[chunk_node.chunk_sentence_id[i]].sentence_text)   
        return ".".join(expand_sentences)

    
async def _answer_query(engine,query:str,search_results:dict)->str:
    context_standard_chunk=[]
    for i,chunk in enumerate(search_results["top_chunk_info"]):
        chunk_node=engine.chunks[chunk[0]]
        chunk_score=chunk[1]
        context_standard_chunk.append(f"传统检索第{i+1}个切块(匹配度：{chunk_score:.2f}):\n{chunk_node.chunk_text}")
    context_hop1=[]
    for i,hop1 in enumerate(search_results["hop1_results"]):
        sentence_node=hop1["sentence"]
        entity_info=";".join([f"{entity.entity_name}({entity.entity_type},重要性{entity.entity_importance})" for entity in hop1["entity"]])
        context_hop1.append(f"一跳信息第{i+1}个句子(匹配度：{hop1['score']:.2f},相关实体：{entity_info}):\n{sentence_node.sentence_text}")

HyperGraph.answer_query= lambda self,query,search_results:_answer_query(self,query,search_results)


async def verify_query(engine,query:str,search_results:dict)->str:
    for hop1 in search_results["hop1_results"]:
        hop1["expand_context"]=engine._expand_context(hop1["sentence"],hop1["sentence"].chunk_id[0])
    return "验证通过"

HyperGraph.verify_query= lambda self,query,search_results:verify_query(self,query,search_results)
