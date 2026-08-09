# RQ6 双人证据链标注使用说明

本说明用于标注 RQ6 的 Physics gold evidence。目标不是给问题写一个
“看起来合理”的答案，而是为每题确定**最少且不可缺少**的原始证据句、它们的
顺序，以及相邻证据之间的 bridge entity。最终将以这些证据链评估
CascHyper-RAG 与 Hyper-RAG 的检索能力。

## 目录分别是什么

| 目录 | 用途 | 是否编辑 |
|---|---|---|
| `annotator_A/packages/` | A 的题目阅读包（Markdown） | 否 |
| `annotator_A/drafts/` | A 的逐题 JSON 标注草稿 | 是 |
| `annotator_B/packages/` | B 的题目阅读包（Markdown） | 否 |
| `annotator_B/drafts/` | B 的逐题 JSON 标注草稿 | 是 |
| `candidate_index.jsonl` | 机器可读的候选句索引，供程序检查或检索 | 否 |

两位标注者只能查看自己的 `packages` 和 `drafts`，不要在独立标注阶段查看或
讨论对方的文件。`packages` 中的 `stage_ref` 和 40 个候选句都只是线索，**不是
gold**；不能因为它排在前面或出现于 `stage_ref` 就直接选中。

## 每题怎么做

以 `physics_s2_001` 为例，打开同名的两个文件：

1. 阅读 `packages/physics_s2_001.md`：先看问题与 Stage，再查看 `stage_ref`
   候选文字和下方表格的 source sentence。
2. 打开并只编辑 `drafts/physics_s2_001.json`。
3. 在候选表中找能直接支持推理链的原始句。复制表格中的 `sentence_id`，不要填
   Rank、document ID、文本片段，也不要填 RAG 内部 ID。
4. 判断该题能否构成完整证据链，然后填“可标注完整链”或“不可标注完整链”之一。
5. 将 `annotation_status` 改为 `complete`，保存 JSON。不要修改
   `question_id` 或 `stage`。

如果 40 条候选句没有所需证据，可以在冻结语料清单
`data/prepared/corpus_manifest.json` 中检索，仍然只能记录其中的 canonical
`sentence_id`。

## 可标注完整链：如何填写

`eligible_for_full_chain` 设为 `true`。只保留必要的 hop：

| Stage | `gold_hops` | `bridges` |
|---:|---|---|
| 1 | 1 个句子 | 空数组 `[]` |
| 2 | 2 个按因果/论证顺序排列的句子 | 1 个连接 Hop 1 与 Hop 2 的实体 |
| 3 | 3 个按顺序排列的句子 | 2 个，分别连接 1→2、2→3 |

“不可缺少”指去掉该句后，问题所需的某一个关键事实或连接关系不再有原文支持。
不要因为句子相关、重复表述、或只提供背景就纳入 gold。一个句子可以同时含有
多个事实，但 `role` 应说明它在该 hop 的核心作用。

Stage 2 的简化填写示意（句子 ID 仅示意，实际以你选择的原文为准）：

```json
{
  "annotation_status": "complete",
  "question_id": "physics_s2_001",
  "stage": 2,
  "eligible_for_full_chain": true,
  "gold_hops": [
    {
      "hop": 1,
      "sentence_id": "physics_doc_007_s_02754",
      "role": "说明环形激光陀螺不抗拒姿态改变的光学原因",
      "required": true
    },
    {
      "hop": 2,
      "sentence_id": "physics_doc_007_s_02755",
      "role": "说明该特性使其适用于剧烈机动的导弹或飞机",
      "required": true
    }
  ],
  "bridges": [
    {
      "canonical_entity": "ring laser gyro",
      "aliases": ["ring laser gyros", "ring laser gyroscope"],
      "from_hop": 1,
      "to_hop": 2
    }
  ],
  "gold_topics": ["physics_doc_007:2011037777"],
  "chain_scope": "intra_chunk",
  "annotation_rationale": "第一句给出原因，第二句给出该原因对应的应用结论；两句均不可缺少。"
}
```

### 字段含义

- `gold_hops[].hop`：必须从 1 开始连续编号，且顺序是论证顺序，不是候选表排名。
- `gold_hops[].sentence_id`：原始、冻结语料中的 canonical ID。
- `gold_hops[].role`：用简短文字说明该句为什么必要。
- `bridges[].canonical_entity`：能把相邻 hop 真正连接起来的规范实体或概念；
  `aliases` 填原句的其他写法。Stage 3 填两个 bridge，分别为 1→2、2→3。
- `gold_topics`：填所选 gold 句所在的 `canonical topic`，去重后保留即可。
- `chain_scope`：所有 gold 句属于同一 `canonical topic` 时填 `intra_chunk`；同一
  `document_id` 但 topic 不同填 `cross_chunk`；出现不同 `document_id` 填
  `cross_document`。
- `annotation_rationale`：说明链条为何成立以及每个证据句为何不可替代；不写模型答案。

## 不可标注完整链：如何填写

当在冻结语料中找不到所需数量的、按顺序相连且不可缺少的原文句时，仍须完成该题：

1. `eligible_for_full_chain` 设为 `false`；
2. **必须**把 `gold_hops` 和 `bridges` 都改为 `[]`；
3. 在 `annotation_rationale` 中简要写明缺失的是哪一跳或哪条连接关系；
4. `annotation_status` 设为 `complete`。

```json
{
  "annotation_status": "complete",
  "question_id": "physics_s2_001",
  "stage": 2,
  "eligible_for_full_chain": false,
  "gold_hops": [],
  "bridges": [],
  "gold_topics": [],
  "chain_scope": "",
  "annotation_rationale": "语料中没有找到能把所需两跳连接起来的不可缺少原文证据。"
}
```

不要把空的 `sentence_id` 或空 bridge 留在 JSON 中；这会导致后续验证失败。

## 独立标注完成后

每位标注者保留自己的 120 个 JSON 草稿，不要覆盖对方目录，也不要手动将两人的
文件拼成最终 gold。下一步会按同一 `question_id` 比较 A/B 的 eligibility、证据句和
bridge，再由专家完成分歧裁决，生成唯一的 `gold_evidence_chains.json`。只有裁决后的
单一 gold 文件才进入 RQ6 指标计算。

## 完成前快速自查

- `annotation_status` 是否为 `complete`？
- `eligible_for_full_chain` 是否是 `true` 或 `false`，而不是 `null`？
- 若为 `true`，hop 数和 bridge 数是否符合该 Stage？
- 若为 `false`，`gold_hops` 与 `bridges` 是否均为 `[]`？
- 每个 `sentence_id` 是否从 canonical 候选表或 corpus manifest 原样复制？
- A/B 是否始终保持彼此独立？
