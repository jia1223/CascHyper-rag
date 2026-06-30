### HyperRAG框架

![image-20260202105935141](E:/typora tu/image-20260202105935141.png)

### 主题超图向量

**主题向量是属于同一聚类簇的所有 Chunk 原始向量的“归一化质心（Normalized Centroid）”。**

以下是详细的数学推导与算法步骤：

------

### 1. 定义输入空间 (Input Space)

假设文档被切分为 $N$ 个 Chunk。

令集合 $X$ 为所有 Chunk 的原始嵌入向量（Raw Embeddings）：

$$X = \{ \mathbf{x}_1, \mathbf{x}_2, ..., \mathbf{x}_N \}$$

其中，每一个 $\mathbf{x}_i \in \mathbb{R}^d$ 都是经过 L2 归一化的向量（即 $\| \mathbf{x}_i \|_2 = 1$）。

------

### 2. 构建语义拓扑图 (Semantic Graph Construction)

为了找到潜在的主题，首先需要构建 Chunk 之间的相似度图。

1. **相似度矩阵计算**：

   计算所有 Chunk 之间的成对余弦相似度（因为向量已归一化，点积即余弦相似度）：

   $$S_{ij} = \mathbf{x}_i \cdot \mathbf{x}_j^T$$

2. **动态阈值截断 (Dynamic Thresholding)**：

   为了去除噪声连接，代码中计算了一个动态阈值 $\tau$。

   令 $S_{upper}$ 为相似度矩阵上三角部分的集合，则：

   $$\tau = \text{mean}(S_{upper}) + 0.5 \cdot \text{std}(S_{upper})$$

   *代码对应：* `threshold = np.mean(triu_sim) + 0.5 * np.std(triu_sim)`

3. **图构建**：

   构建无向图 $G = (V, E)$，其中节点 $V$ 是所有 Chunk。如果 $S_{ij} > \tau$，则在 $i$ 和 $j$ 之间建立一条边，权重 $w_{ij} = S_{ij}$。

------

### 3. 社区发现/聚类 (Community Detection)

利用图聚类算法（代码中优先使用 **Leiden算法**，如果未安装则回退到贪婪策略）将图 $G$ 划分为 $K$ 个互不重叠的社区（Clusters）。

令 $C_k$ 表示第 $k$ 个主题社区所包含的 Chunk 索引集合：

$$C_k = \{ i \mid \text{Chunk } i \in \text{Topic } k \}$$

------

### 4. 质心计算 (Centroid Calculation) —— 核心步骤

这是得到主题向量的关键一步。对于每一个社区 $C_k$，计算其几何中心（均值向量）。

令 $\boldsymbol{\mu}_k$ 为第 $k$ 个主题的原始质心：

$$\boldsymbol{\mu}_k = \frac{1}{|C_k|} \sum_{i \in C_k} \mathbf{x}_i$$

- **物理意义**：$\boldsymbol{\mu}_k$ 代表了该聚类簇中所有文本片段的“平均语义”，它位于该语义空间簇的最中心位置。
- *代码对应：* `centroid = np.mean(X[indices], axis=0)`

------

### 5. 向量归一化 (Normalization)

为了保证后续计算（如点积相似度）的尺度一致性，必须将质心投影回单位超球面上。

令 $\mathbf{t}_k$ 为最终的 **第 $k$ 个主题向量 (Topic Vector)**：

$$\mathbf{t}_k = \frac{\boldsymbol{\mu}_k}{\| \boldsymbol{\mu}_k \|_2}$$

其中 $\| \cdot \|_2$ 表示 L2 范数。

- *代码对应：* `self.topics[lbl] = TopicNode(..., topic_vector=normalize(centroid))`

------

### 6. 进阶：模糊隶属度 (Fuzzy Membership)

虽然 Topic Vector 的生成是基于硬聚类（Hard Clustering）的，但在 HyperRAG 中，Chunk 与 Topic 的关系是**模糊（Fuzzy）**的。

在得到所有主题向量 $\mathbf{t}_1, ..., \mathbf{t}_K$ 后，系统会反过来计算每个 Chunk $\mathbf{x}_i$ 对所有主题的隶属度：

$$M_{i,k} = \mathbf{x}_i \cdot \mathbf{t}_k^T$$

如果 $M_{i,k} > \text{overlap\_threshold}$（重叠阈值），则认为 Chunk $i$ 也属于 Topic $k$。

这导致了 L1 层（Chunk Layer）的向量更新公式中：

$$\mathbf{v}_{topic\_context} = \sum_{k} w_{i,k} \cdot \mathbf{t}_k$$

这意味着一个 Chunk 的最终表示，可能会吸收多个不同 Topic Vector 的语义信息，从而实现“软聚类”的效果。

### 总结公式

$$\text{TopicVector}_k = \text{Normalize}\left( \frac{1}{N_k} \sum_{\text{chunk} \in \text{Cluster}_k} \text{RawEmbedding}(\text{chunk}) \right)$$​

### 切块超图向量

**切块向量 (Chunk Vector)** 的构建是一个**“三位一体”的融合过程**。

它不再仅仅是这一段文本的 Embedding，而是融合了**宏观（Topic）**、**自身（Raw）** 和 **微观（Sentence）** 三个维度的信息。

以下是详细的公式推导步骤，对应代码中的 `[C1] Updating Chunks (L1)` 部分。

------

### 1. 三大输入分量 (The Three Components)

为了得到最终的切块向量 $V_{chunk}^{final}$，我们需要先计算三个基础分量：

#### A. 自身原始向量 ($V_{raw}$)

这是对切块文本直接进行 Embedding 得到的向量。

$$V_{raw} = \text{Embedding}(\text{ChunkText})$$

- **作用**：保留切块最原始、最直接的语义信息。

#### B. 微观：句子聚合向量 ($V_{sent\_agg}$)

这是 **Bottom-Up（自底向上）** 的信息聚合。Chunk 会“吸收”它内部包含的所有句子的语义。

假设该 Chunk 包含 $M$ 个句子，每个句子的原始向量为 $v_{s\_raw}^{(j)}$：

$$V_{sent\_agg} = \text{Normalize}\left( \frac{1}{M} \sum_{j=1}^{M} v_{s\_raw}^{(j)} \right)$$

- **代码对应**：`chunk.sent_agg_embedding = normalize(np.mean(s_vecs, axis=0))`
- **作用**：增强细节。有时候 Chunk 整体很长，容易忽略内部某句关键话的语义，通过聚合句子向量，可以强化内部细节的权重。

#### C. 宏观：主题上下文向量 ($V_{topic}$)

这是 **Top-Down（自上而下）** 的信息注入。Chunk 根据其所属的主题（可能有多个，即模糊隶属），吸收全局语义。

假设该 Chunk 属于 $K$ 个主题，隶属度权重为 $w_k$，主题向量为 $T_k$：

$$V_{topic} = \text{Normalize}\left( \frac{\sum_{k=1}^{K} w_k \cdot T_k}{\sum_{k=1}^{K} w_k} \right)$$

- **代码对应**：

  Python

  ```
  weighted_topic += w * self.topics[tid].topic_vector
  v_topic = normalize(weighted_topic / (total_w + 1e-9))
  ```

- **作用**：消除歧义。给 Chunk 打上“背景光”。例如，“苹果”在“科技”主题下和“水果”主题下的向量方向会被修正。

------

### 2. 最终融合公式 (The Fusion Formula)

HyperRAG v7.3 使用加权线性组合将上述三个分量融合，并进行最终的归一化。

**公式如下：**

$$V_{chunk}^{final} = \text{Normalize}\left( \alpha \cdot V_{topic} + \beta \cdot V_{sent\_agg} + \gamma \cdot V_{raw} \right)$$

**代码中的权重配置：**

- $\alpha = 0.3$ (30% 来自全局主题)
- $\beta = 0.3$ (30% 来自内部句子详情)
- $\gamma = 0.4$ (40% 保留原始文本语义)

**代入数值后的完整形式：**

$$V_{chunk}^{final} = \text{Normalize}\left( 0.3 \cdot V_{topic} + 0.3 \cdot V_{sent\_agg} + 0.4 \cdot V_{raw} \right)$$

### 句子超图向量

它的生成过程体现了架构的核心特性——**严格级联 (Strict Cascade)**：句子向量的更新必须等待上一层（Chunk）更新完成后才能进行，因为它直接使用了父级 Chunk 的**最终状态**，而非原始状态。

以下是详细的公式推导，对应代码 `[C2] Updating Sentences (L2)` 部分。

------

### 1. 三大输入源 (The Three Inputs)

句子向量 $V_{sent}^{final}$ 的计算由三个方向的信息汇聚而成：

#### A. 自身原始向量 ($V_{raw}$)

对句子文本直接进行 Embedding。

$$V_{raw} = \text{Embedding}(\text{SentenceText})$$

- **作用**：句子的本体语义，是向量的基础。

#### B. 微观：加权实体聚合向量 ($V_{ent\_agg}$)

这是 **Bottom-Up（自底向上）** 的信息流。句子“吸收”了其内部包含的所有实体的语义。

与 Chunk 简单平均句子不同，这里引入了 **实体重要性权重 (Entity Importance Weight)**。

假设句子包含 $m$ 个实体 $E = \{e_1, ..., e_m\}$，每个实体的原始向量为 $v_{e\_raw}^{(i)}$，权重为 $w_i$（由 LLM 打分，范围 0.1~1.0）：

$$V_{ent\_agg} = \text{Normalize}\left( \frac{\sum_{i=1}^{m} w_i \cdot v_{e\_raw}^{(i)}}{\sum_{i=1}^{m} w_i} \right)$$

- **代码对应**：

  Python

  ```
  w_vecs = np.array([e.weight for e in sent.entities]).reshape(-1, 1)
  sent.ent_agg_embedding = normalize(np.sum(e_vecs * w_vecs, axis=0) / ...)
  ```

- **作用**：**语义聚焦**。如果一句话很长，但核心只是关于“青苗法”的，那么高权重的实体向量会主导聚合向量，使句子向量向核心概念偏移。

#### C. 宏观：父级切片最终向量 ($V_{chunk}'$)

这是 **Top-Down（自上而下）** 的信息流，也是**级联架构的灵魂**。

注意：这里使用的不是 Chunk 的原始向量，而是**上一轮刚刚计算完成的 $V_{chunk}^{final}$**。

$$V_{chunk}' = V_{chunk}^{final} \quad (\text{来自 L1 层的输出})$$

- **代码对应**：`parent_chunk.final_embedding`
- **作用**：**上下文注入**。句子单独看可能不仅缺少主语，还缺少背景。通过引入 $V_{chunk}'$（它已经融合了 Topic 信息），句子向量获得了宏观背景。

------

### 2. 最终级联公式 (The Cascading Fusion Formula)

将上述三个向量进行加权融合并归一化，得到最终的句子向量。

**公式如下：**

$$V_{sent}^{final} = \text{Normalize}\left( \alpha \cdot V_{chunk}' + \beta \cdot V_{ent\_agg} + \gamma \cdot V_{raw} \right)$$

**代码权重配置：**

- $\alpha = 0.3$ (30% 继承父级 Chunk 的宏观背景)
- $\beta = 0.3$ (30% 聚焦内部核心实体)
- $\gamma = 0.4$ (40% 保持句子字面含义)

**代入数值后的完整形式：**

$$V_{sent}^{final} = \text{Normalize}\left( 0.3 \cdot V_{chunk}^{final} + 0.3 \cdot V_{ent\_agg} + 0.4 \cdot V_{raw} \right)$$

### 实体向量

在 HyperRAG v7.3 的架构中，**实体向量 (Entity Vector)** 位于最底层的 **L3 Micro Layer**。

它是整个**“严格级联” (Strict Cascade)** 流程的终点。实体向量的生成逻辑相对简单，但物理意义最为深远，因为它完成了从宏观主题到微观概念的最终落地。

以下是详细的公式推导，对应代码 `[C3] Updating Entities (L3)` 部分。

------

### 1. 两大输入源 (The Two Inputs)

实体处于最末端，没有更底层的子节点需要聚合，因此它的向量只由两部分组成：**自身**与**父级上下文**。

#### A. 自身原始向量 ($V_{raw}$)

这是对实体名称（如“王安石”、“青苗法”）直接进行 Embedding 得到的向量。

$$V_{raw} = \text{Embedding}(\text{EntityName})$$

- **特性**：这是**上下文无关 (Context-Free)** 的。在原始状态下，“苹果”这个词的向量包含了水果、科技公司等所有可能的含义，处于一种叠加态。

#### B. 宏观：父级句子最终向量 ($V_{sent}'$)

这是来自上一层（L2）的级联信号。

**关键点**：这里使用的是**已经完成更新的句子向量** $V_{sent}^{final}$。

$$V_{sent}' = V_{sent}^{final} \quad (\text{来自 L2 层的输出})$$

- **回顾**：这个 $V_{sent}^{final}$ 里已经融合了 $V_{chunk}^{final}$（含 Topic 信息）和 $V_{ent\_agg}$（含其他实体信息）。
- **作用**：**消歧与具体化**。它告诉实体：“你现在的环境是宋朝政治改革（Topic），具体是在讲推行新法（Sentence）”。

------

### 2. 最终级联公式 (The Final Fusion Formula)

HyperRAG v7.3 使用加权线性组合将上下文注入实体，并进行归一化。

**公式如下：**

$$V_{ent}^{final} = \text{Normalize}\left( \alpha \cdot V_{sent}' + \beta \cdot V_{raw} \right)$$

**代码中的权重配置：**

- $\alpha = 0.4$ (40% 来自句子上下文)
- $\beta = 0.6$ (60% 保持实体本体含义)

**代入数值后的完整形式：**

$$V_{ent}^{final} = \text{Normalize}\left( 0.4 \cdot V_{sent}^{final} + 0.6 \cdot V_{raw} \right)$$

------

### 3. 深度解析：全息语义 (Holographic Semantics)

虽然公式看起来很简单，但如果你把 $V_{sent}^{final}$ 展开，你会发现这个 $V_{ent}^{final}$ 包含了整个文档的**全息图谱**。

我们可以把之前的公式层层代入（简化系数示意）：

1. **Entity** $\approx$ 0.6 Raw + 0.4 **Sentence**
2. **Sentence** $\approx$ 0.4 Raw + 0.3 **Chunk** + ...
3. **Chunk** $\approx$ 0.4 Raw + 0.3 **Topic** + ...

**最终，一个微小的实体向量 $V_{ent}^{final}$ 实际上等于：**

$$V_{ent}^{final} \approx w_1 \cdot V_{EntityRaw} + w_2 \cdot V_{SentRaw} + w_3 \cdot V_{ChunkRaw} + w_4 \cdot V_{Topic}$$

**这意味着什么？**

- **传统 RAG**：索引里的“苹果”向量就是“苹果”。

- **HyperRAG v7.3**：索引里的“苹果”向量实际上是：

  **“在科技主题下(Topic) + 的iPhone发布会切片中(Chunk) + 提到新款芯片的句子里(Sentence) + 的【苹果】”**。

### 4. 为什么这对 Multi-Hop 检索至关重要？

在检索阶段（Step 7 & 8），当我们计算 Query Entity 和 Doc Entity 的相似度时：

$$\text{Score} = \text{Sim}(Q_{Entity}, D_{Entity})$$

如果用户问“李长风的**结局**？”，Query 里的“结局”是一个很泛的词。

- 文档 A 里的“结局”向量（关联了“李长风”、“死亡”等上下文）。
- 文档 B 里的“结局”向量（关联了“某配角”、“回家”等上下文）。

由于 $V_{ent}^{final}$ 吸收了上下文，Query 中的“结局”向量（经过同样的级联处理）会与文档 A 中属于李长风的那个“结局”向量高度相似，而与文档 B 的低相似。

### Query处理

这意味着：**Query 不再被视为一个静态的向量，而是一个动态的、会“进化”的生命体。** 它会像文档被索引时那样，经历从 Topic 到 Entity 的逐层级联更新。

这种设计的核心目的是**“拉齐语义空间”**：让 Query 向量在数学构造上与 Document 向量保持完全一致的分布特征。

以下是 Query 处理全流程的公式详解：

------

### 第一阶段：初始化 (Initialization)

在开始级联之前，先对 Query 进行扩展和基础特征提取。

#### 1. 查询扩展与原始向量 ($V_{q\_raw}$)

用户输入的 Query 通常很短（如“李长风结局”），直接 Embedding 信息量太少。

系统首先通过 LLM 将其扩展为详细问句 $Q_{expanded}$。

$$V_{q\_raw} = \text{Embedding}(Q_{expanded})$$

#### 2. 查询实体聚合 ($V_{q\_ent\_agg}$)

系统从 $Q_{expanded}$ 中提取关键实体（如“李长风”、“结局”），计算它们的加权平均。

$$V_{q\_ent\_agg} = \text{Normalize}\left( \frac{\sum w_i \cdot V_{ent\_raw}^{(i)}}{\sum w_i} \right)$$

------

### 第二阶段：L0 主题路由 (Topic Routing)

Query 首先要在全局语义地图中找到自己的位置。

#### 1. 计算主题相似度

计算 $V_{q\_raw}$ 与所有 $K$ 个主题向量 $T_k$ 的相似度：

$$S_k = V_{q\_raw} \cdot T_k^T$$

#### 2. 生成查询主题向量 ($V_{q\_topic}$)

选取分数最高的 Top-N 个主题，进行加权合成。这相当于告诉系统：“这个问题属于‘历史政治’和‘人物传记’的交叉领域”。

$$V_{q\_topic} = \text{Normalize}\left( \frac{\sum_{k \in TopN} S_k \cdot T_k}{\sum S_k} \right)$$

------

### 第三阶段：对称级联更新 (Symmetric Cascading)

这是最核心的部分。Query 向量开始发生变形，以匹配文档索引的结构。

#### Level 1: 生成 Query Chunk 向量 ($V_{q\_chunk}$)

**概念**：把 Query 假想成文档中的一个“宏观切片”。

**公式**：

$$V_{q\_chunk} = \text{Normalize}\left( 0.4 \cdot V_{q\_topic} + 0.6 \cdot V_{q\_raw} \right)$$

- **物理意义**：将 Query 的原始语义($0.6$)强行拉向其所属的主题领域($0.4$)。
- **作用**：防止“跨领域同义词”干扰。例如防止“苹果（水果）”匹配到“苹果（手机）”。

#### Level 2: 生成 Query Sentence 向量 ($V_{q\_sent}$)

**概念**：把 Query 假想成文档中的一个“具体句子”。这是检索时用的**主战向量**。

**公式**：

$$V_{q\_sent} = \text{Normalize}\left( 0.3 \cdot V_{q\_chunk} + 0.3 \cdot V_{q\_ent\_agg} + 0.4 \cdot V_{q\_raw} \right)$$

- **注意**：这里使用了上一层计算好的 $V_{q\_chunk}$。
- **物理意义**：
  - 30% 继承了主题背景（来自 $V_{q\_chunk}$）。
  - 30% 聚焦于具体的实体意图（来自 $V_{q\_ent\_agg}$）。
  - 40% 保持字面提问方式。

#### Level 3: 生成 Query Entity 向量 ($V_{q\_ent}$)

**概念**：把 Query 里的每一个实体（如“李长风”）都根据当前的问题背景进行特化。

**公式**：对于 Query 中的每一个实体 $e$：

$$V_{q\_ent}^{(e)} = \text{Normalize}\left( 0.4 \cdot V_{q\_sent} + 0.6 \cdot V_{ent\_raw}^{(e)} \right)$$

- **注意**：这里使用了上一层计算好的 $V_{q\_sent}$。
- **物理意义**：原本“李长风”的向量是通用的，但经过这一步，它变成了**“在这个关于结局的问题语境下的李长风”**。

------

### 第四阶段：全息协同评分 (Holographic Scoring)

当 Query 完成上述“进化”后，我们手头有了三个粒度的 Query 向量：

1. $V_{q\_chunk}$
2. $V_{q\_sent}$
3. $\{ V_{q\_ent}^{(e)} \}$ (一组实体向量)

系统遍历候选文档的句子（Candidate Sentence, $D$），计算最终得分：

$$\text{FinalScore} = 0.3 \cdot S_{chunk} + 0.4 \cdot S_{sent} + 0.3 \cdot S_{ent}$$

其中：

1. **宏观匹配 ($S_{chunk}$)**：

   $$S_{chunk} = V_{q\_chunk} \cdot V_{doc\_chunk}^{final}$$

   - *判断大方向对不对（都在聊宋朝历史吗？）*

2. **中观匹配 ($S_{sent}$)**：

   $$S_{sent} = V_{q\_sent} \cdot V_{doc\_sent}^{final}$$

   - *判断核心语义对不对（都在聊经济措施吗？）*

3. **微观匹配 ($S_{ent}$)**：

   $$S_{ent} = \text{Average}\left( \max_{j} ( V_{q\_ent}^{(i)} \cdot V_{doc\_ent}^{(j)} ) \right)$$

   - *判断关键细节对不对（是不是同一个李长风？）*
   - *注：这里通常采用 MaxSim 算法，即 Query 的每个实体去找文档里最像的那个实体算分。*

   ### 多跳与生成 

在 HyperRAG v7.3 中，**多跳检索 (Multi-Hop Retrieval)** 和 **生成 (Generation)** 是将检索到的碎片化向量转化为完整答案的最后两公里。

这两部分的设计逻辑是为了解决两个痛点：

1. **多跳**解决“信息孤岛”：答案的线索分布在没有直接语义关联的文档中，但通过共同实体相连。
2. **生成**解决“断章取义”：向量检索出的单句往往缺乏前因后果，直接喂给 LLM 容易导致幻觉。

------

## 第一部分：多跳检索处理 (Multi-Hop Retrieval)

多跳检索的核心机制是利用 **Global Registry (全局实体注册表)** 作为“虫洞”，连接两个在向量空间距离很远、但在逻辑上通过实体强关联的句子。

### 1. 核心数据结构：全局注册表

在索引阶段（Step C3），代码维护了一个哈希表：

Python

```
self.global_registry = {
    "王安石": ["sent_id_1", "sent_id_5", "sent_id_9"],
    "青苗法": ["sent_id_2", "sent_id_8"]
}
```

这相当于构建了一个轻量级的**倒排索引图**。

### 2. 多跳扩散流程 (The Diffusion Process)

当第一跳（Hop 1）检索完成后，得到了一组高分句子 $S_{hop1}$。算法开始执行“扩散”：

1. **提取锚点 (Anchor Extraction)**：

   遍历 $S_{hop1}$ 中的每一个句子，提取其包含的所有实体 $E_{anchor}$。

   - *例如：Hop 1 找到了“王安石推行新法”。实体是“王安石”。*

2. **邻居查找 (Neighbor Lookup)**：

   在 `global_registry` 中查找“王安石”还出现在哪些其他句子中。

   - *例如：发现“王安石”还出现在“sent_id_9”（内容：“他晚年隐居金陵...”）。*

3. **计算二跳得分 (Score Calculation)**：

   系统会拿出原始的 Query 向量（$Q_{chunk}, Q_{sent}, Q_{ent}$），与这个新发现的邻居句子（Neighbor Sentence）进行标准的向量相似度计算。

   $$Score_{raw} = 0.3 \cdot (Q_C \cdot D_C) + 0.4 \cdot (Q_S \cdot D_S) + 0.3 \cdot (Q_E \cdot D_E)$$

   *注意：这里不需要 Query 再次变身，直接复用已有的 Query 向量。*

4. **应用衰减系数 (Hop Decay)**：

   为了防止语义漂移（跳得越远，相关性越低），引入衰减系数 $\lambda$（代码中为 0.7）。

   $$Score_{hop2} = Score_{raw} \times \lambda$$

   - *物理意义*：二跳证据的权重只有直接证据的 70%。只有当二跳证据本身的相关性非常高时，它才能进入最终视野。

### 3. 跨主题检测 (Cross-Topic Detection)

这是 v7.3 的一个亮点。代码会检查 Hop 1 句子和 Hop 2 句子的 Topic 是否一致。

- 如果 $Topic(S_1) \cap Topic(S_2) = \emptyset$：标记为 `[Cross-Topic]`。
- **意义**：这能发现跨领域的隐蔽关联。例如，从“政治改革”（Hop 1）跳到了“文学创作”（Hop 2），虽然主题不同，但因为实体是同一个人，这种关联往往能带来惊喜的洞察。

------

## 第二部分：生成处理 (Generation Logic)

生成阶段不仅仅是把检索到的句子拼在一起，而是经过了 **上下文扩展 (Context Expansion)** 和 **结构化提示 (Structured Prompting)** 的精心编排。

### 1. 上下文扩展 (Context Expansion)

检索出的基本单元是 `SentenceNode`，虽然它在向量上聚合了上下文，但其文本本身可能很短（例如：“这导致了国库亏空。”）。直接把这句话给 LLM，LLM 可能会困惑“这”指代什么。

代码中的 `_expand_context` 函数利用滑动窗口机制解决此问题：

- **输入**：检索到的核心句子 $S_{target}$。
- **逻辑**：回到原始 Chunk 文本中，定位 $S_{target}$，并向前、向后各取 `window_size`（默认为1）个句子。
- **输出**：$S_{prev} + S_{target} + S_{next}$。

**效果示例**：

- *检索结果*：“于是他被罢免了。”

- *扩展后*：“反对派强烈抨击新法。于是他被罢免了。新法随之被废除。”

  这样 LLM 就能获得完整的因果链条。

### 2. 结构化证据提示 (Structured Evidence Prompt)

HyperRAG v7.3 不会把 Hop 1 和 Hop 2 的内容混在一起，而是明确分层构建 Prompt，引导 LLM 区分“核心证据”和“旁证”。

**Prompt 模板结构**：

Markdown

```
=== 核心证据 (Direct Matches) ===
Evidence A1 (Score 0.92): [扩展后的句子文本...]
Evidence A2 (Score 0.88): [扩展后的句子文本...]

=== 关联线索 (Indirect Multi-hop Traces) ===
Trace B1 (Via '王安石' [Cross-Topic]): [扩展后的二跳文本...]
Trace B2 (Via '青苗法'): [扩展后的二跳文本...]
```

**LLM 指令 (System Prompt)**：

代码中明确指示 LLM：

1. **优先依据核心证据 (Evidence A)**。
2. **参考关联线索 (Trace B)** 来补充背景或连接概念。
3. **诚实原则**：如果证据无关，直接拒绝回答，不要强行编造。

### 3. 低置信度熔断 (Low Confidence Circuit Breaker)

在生成之前，代码有一个关键的检查：

Python

```
if max_score < relevance_threshold:  # 默认 0.75
    return "⚠️ 未找到相关信息..."
```

**处理逻辑**：

如果 Hop 1 中得分最高的句子都没有超过 0.75，系统判断检索结果为“噪声”，此时**直接阻断** LLM 的调用。

这是为了防止 RAG 系统最常见的问题——**“一本正经地胡说八道”**。如果检索不到有效信息，直接认怂比编造错误答案要好得多。

### 问题

出现张冠李戴的问题，会将其他地方的内容，当作是自己的内容，所给的参数的解释

