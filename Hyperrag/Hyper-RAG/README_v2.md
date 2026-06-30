# 新版超图RAG系统使用说明

## 概述

这是基于用户图片思路重新设计的超图RAG系统，主要实现了以下功能：

1. **文本切块** - 将输入文本分割成合适大小的块，保持句子完整性
2. **句子级实体提取** - 对每个句子进行多个实体提取，使用jieba分词
3. **实体聚合与超边构建** - 将相关实体聚合形成超边，支持共现和语义关联
4. **问题实体提取** - 对输入问题提取实体并构建超边
5. **超边匹配检索** - 使用问题超边匹配文档超边进行相关性检索

## 系统架构

### 核心类

- **Entity**: 实体类，包含名称、类型、描述等信息
- **Hyperedge**: 超边类，连接多个实体，包含权重和描述
- **TextChunk**: 文本块类，包含内容、句子、实体和超边
- **HypergraphRAGv2**: 主要系统类，实现所有核心功能

### 工作流程

```
输入文本 → 文本切块 → 句子分割 → 实体提取 → 超边构建 → 存储
                                                    ↓
查询问题 → 实体提取 → 超边构建 → 超边匹配 → 相关文档检索
```

## 安装依赖

```bash
pip install -r requirements_v2.txt
```

## 使用方法

### 基本使用

```python
from hypergraph_rag_v2 import HypergraphRAGv2

# 初始化系统
rag_system = HypergraphRAGv2(
    chunk_size=500,        # 文本块大小
    chunk_overlap=50       # 文本块重叠大小
)

# 处理文档
text = "你的文档内容..."
chunks = rag_system.text_chunking(text)
rag_system.extract_entities_from_chunks()
rag_system.build_hyperedges_from_chunks()

# 查询
query = "你的问题"
results = rag_system.retrieve_relevant_chunks(query, top_k=3)

# 查看结果
for result in results:
    print(f"相似度: {result['score']}")
    print(f"内容: {result['content']}")
    print(f"相关实体: {result['entities']}")
```

### 详细步骤

#### 1. 文本切块

```python
# 将长文本分割成合适大小的块
chunks = rag_system.text_chunking(text)
print(f"生成了 {len(chunks)} 个文本块")
```

#### 2. 实体提取

```python
# 从所有文本块中提取实体
chunk_entities = rag_system.extract_entities_from_chunks()
print(f"提取了 {len(rag_system.entities)} 个唯一实体")
```

#### 3. 超边构建

```python
# 构建超边连接相关实体
hyperedges = rag_system.build_hyperedges_from_chunks()
print(f"构建了 {len(hyperedges)} 个超边")
```

#### 4. 查询检索

```python
# 检索相关文档
query = "机器学习是什么？"
results = rag_system.retrieve_relevant_chunks(query, top_k=3)
```

#### 5. 保存和加载

```python
# 保存超图结构
rag_system.save_hypergraph("my_hypergraph.json")
```

## 核心特性

### 1. 智能文本切块

- 按句子边界切块，保持语义完整性
- 支持重叠切块，避免信息丢失
- 自动处理中文句子分割

### 2. 多层次实体提取

- 使用jieba分词进行中文实体识别
- 支持多种实体类型：人名、地名、机构名、概念等
- 句子级别的精细化实体提取

### 3. 超边构建策略

- **句子内共现超边**: 同一句子中的实体形成超边
- **语义关联超边**: 相同类型的实体形成语义超边
- **权重计算**: 基于实体关系强度分配权重

### 4. 智能匹配算法

- **Jaccard相似度**: 基于实体交集计算相似度
- **语义相似度**: 使用嵌入模型计算语义相似度
- **综合评分**: 结合多种相似度指标

## 配置参数

### 初始化参数

- `chunk_size`: 文本块大小（默认500字符）
- `chunk_overlap`: 文本块重叠大小（默认50字符）
- `embedding_model_path`: 嵌入模型路径

### 实体提取配置

系统支持以下实体类型：
- 人名 (nr)
- 地名 (ns) 
- 机构名 (nt)
- 其他专名 (nz)
- 一般名词 (n)
- 动词 (v)
- 形容词 (a)

### 超边类型

- `sentence_cooccurrence`: 句子内实体共现
- `semantic_relation`: 语义关联
- `query_cooccurrence`: 查询实体共现
- `query_semantic`: 查询语义关联

## 性能优化建议

1. **文本预处理**: 清理无关字符和噪声
2. **块大小调整**: 根据文档特点调整块大小
3. **实体过滤**: 过滤低质量实体提升精度
4. **嵌入模型**: 使用更好的中文嵌入模型

## 示例输出

### 实体提取结果
```
chunk_0: ['人工智能', '计算机科学', '智能', '机器']
chunk_1: ['机器学习', '人工智能', '算法', '数据', '决策']
```

### 超边构建结果
```
超边 0: {'人工智能', '计算机科学'} - 句子内实体共现
超边 1: {'机器学习', '算法'} - 句子内实体共现
```

### 检索结果
```
结果 1 (相似度: 0.856):
内容: 机器学习是人工智能的一个重要分支，通过算法使机器能够从数据中学习...
相关实体: ['机器学习', '人工智能', '算法', '数据']
```

## 扩展功能

### 自定义实体提取器

```python
def custom_entity_extractor(sentence):
    # 自定义实体提取逻辑
    entities = []
    # ... 你的实体提取代码
    return entities

# 替换默认提取器
rag_system._extract_entities_from_sentence = custom_entity_extractor
```

### 自定义相似度计算

```python
def custom_similarity(edge1, edge2):
    # 自定义相似度计算逻辑
    return similarity_score

# 替换默认相似度计算
rag_system._calculate_hyperedge_similarity = custom_similarity
```

## 注意事项

1. **内存使用**: 大文档可能消耗较多内存
2. **处理时间**: 实体提取和超边构建需要一定时间
3. **模型依赖**: 嵌入模型需要网络下载
4. **中文支持**: 针对中文文本优化，其他语言可能需要调整

## 故障排除

### 常见问题

1. **jieba分词错误**: 确保正确安装jieba包
2. **嵌入模型加载失败**: 检查网络连接或使用本地模型
3. **内存不足**: 减小chunk_size或分批处理
4. **实体提取质量低**: 调整实体类型过滤条件

### 调试模式

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 更新日志

### v2.0.0
- 重新设计的超图RAG架构
- 基于用户图片思路的实现
- 支持中文文本处理
- 多层次实体提取和超边构建
- 智能超边匹配算法
