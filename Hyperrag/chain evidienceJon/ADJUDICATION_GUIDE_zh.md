# RQ6 双人 Gold 证据链仲裁说明

本说明用于处理 Physics RQ6 双人独立标注完成后的分歧。目标是形成一份唯一、可复现、可验证的最终 gold evidence chain，而不是由仲裁者重新做一遍检索或比较模型答案。

## 仲裁范围与原则

- 仅仲裁 `data/adjudication/packages/` 中的题目。本批次共 92 题。
- A/B 原始标注位于 `data/annotation_packages/annotator_A/drafts/` 与 `annotator_B/drafts/`，**只读，不修改**。
- 仲裁依据只能是冻结语料的 canonical source sentence；不得使用 RAG 输出、最终答案或模型推理结果倒推证据。
- 每题只编辑与其同名的 `data/adjudication/decisions/<question_id>.json`。
- 证据应是回答该题所需的最少且不可缺少的句子；相关背景句、同义重复句不应加入 gold。

## 文件对应关系

以 `physics_s2_046` 为例：

| 文件 | 作用 | 是否编辑 |
|---|---|---|
| `packages/physics_s2_046.md` | 展示问题、A/B 原始记录和双方选中的 canonical 原文句 | 否 |
| `decisions/physics_s2_046.json` | 记录仲裁决定及理由 | 是 |
| `adjudication_queue.jsonl` | 程序可读的完整分歧队列 | 否 |
| `adjudication_manifest.json` | 本批次分歧题清单 | 否 |

仲裁包中的 A/B 标注用于对照，不能再回写到任一标注者目录。

## 每题仲裁步骤

1. 打开同名的 Markdown 仲裁包，阅读题目和 `Difference fields`。
2. 对照 A/B 的 `gold_hops`、`bridges`、`gold_topics` 以及 `annotation_rationale`。
3. 阅读包中列出的 canonical source sentence，判断哪个链条真正按顺序支持问题。
4. 打开同名的 JSON 决议文件，填写 `decision`、必要时填写 `final_gold`，并写明 `adjudication_rationale`。
5. 完成后把 `decision_status` 改为 `complete`，保存为合法 UTF-8 JSON。

`question_id`、`stage`、`difference_fields` 和 `allowed_decisions` 是生成时的元数据，不能修改。

## 四种决议如何选择

### `use_A`

选择 A 的原始标注作为最终 gold。当 A 的 evidence chain、bridge 和 topic 都正确，且无需改动时使用。

```json
{
  "decision_status": "complete",
  "decision": "use_A",
  "final_gold": null,
  "adjudication_rationale": "A 的两跳顺序与原文因果关系一致；B 的第二跳未直接支持问题所需结论。"
}
```

### `use_B`

选择 B 的原始标注作为最终 gold。当 B 的整条链正确且无需改动时使用。

```json
{
  "decision_status": "complete",
  "decision": "use_B",
  "final_gold": null,
  "adjudication_rationale": "B 选择的 bridge entity 与相邻两跳均直接对应，A 的 bridge 只是背景概念。"
}
```

### `revised`

当 A、B 都不应原样采用，或应组合两者的正确部分时使用。必须在 `final_gold` 填写完整、可直接进入最终 gold 文件的对象。`final_gold` 的字段结构与 A/B 原始标注相同，但不需要 `annotation_status`。

Stage 2 示例：

```json
{
  "decision_status": "complete",
  "decision": "revised",
  "final_gold": {
    "question_id": "physics_s2_046",
    "stage": 2,
    "eligible_for_full_chain": true,
    "gold_hops": [
      {
        "hop": 1,
        "sentence_id": "physics_doc_015_s_01999",
        "role": "给出理想主义解决中观察主体意识选择猫的叠加态分支这一关键事实",
        "required": true
      },
      {
        "hop": 2,
        "sentence_id": "physics_doc_015_s_03449",
        "role": "给出自指与纠缠层级在脑—心系统中出现的语境",
        "required": true
      }
    ],
    "bridges": [
      {
        "canonical_entity": "self-consciousness",
        "aliases": ["consciousness of the observing subject"],
        "from_hop": 1,
        "to_hop": 2
      }
    ],
    "gold_topics": ["physics_doc_015:375"],
    "chain_scope": "intra_chunk",
    "annotation_rationale": "两句分别给出观察意识的选择作用和纠缠层级/自指的脑—心语境，按论证顺序共同不可缺少。"
  },
  "adjudication_rationale": "采用 A 的第一跳与 B 所依据段落中的第二跳，并将 bridge 规范化为 self-consciousness。"
}
```

请以当前仲裁包内的真实句子为准；上例只说明 JSON 结构，不代表对 `physics_s2_046` 的预先裁决。

### `ineligible`

当冻结语料无法支撑该题需要的完整、有序链条时使用。此时 `final_gold` 也必须填写，且 `eligible_for_full_chain` 为 `false`，并且 `gold_hops` 与 `bridges` 必须都是空数组。

```json
{
  "decision_status": "complete",
  "decision": "ineligible",
  "final_gold": {
    "question_id": "physics_s2_046",
    "stage": 2,
    "eligible_for_full_chain": false,
    "gold_hops": [],
    "bridges": [],
    "gold_topics": [],
    "chain_scope": "",
    "annotation_rationale": "冻结语料中未找到能按顺序连接所需两跳的不可缺少原文证据。"
  },
  "adjudication_rationale": "A/B 均无法形成可验证的完整链，因此排除该题的 full-chain 计分资格。"
}
```

## 链条规则速查

| Stage | 完整链要求 |
|---:|---|
| 1 | 1 个 indispensable sentence；不需要 bridge。 |
| 2 | 2 个按论证顺序排列的句子；1 个连接 Hop 1 与 Hop 2 的 bridge。 |
| 3 | 3 个按论证顺序排列的句子；2 个 bridge，分别连接 1→2、2→3。 |

bridge 必须是实际连接相邻证据的实体或概念，不能只是两个句子都出现过的宽泛主题词。`canonical_entity` 使用稳定、简洁的规范名称；原文的不同写法放入 `aliases`。

## 提交前检查

- `decision_status` 是否为 `complete`？
- `decision` 是否是四个允许值之一？
- `use_A` 或 `use_B` 时，`final_gold` 是否保持 `null`？
- `revised` 或 `ineligible` 时，`final_gold` 是否已完整填写？
- 是否保留了原始的 `question_id` 与 `stage`？
- 若 `eligible_for_full_chain` 为 `false`，`gold_hops` 和 `bridges` 是否均为 `[]`？
- 每个 `sentence_id` 是否来自冻结 canonical manifest？
- `adjudication_rationale` 是否说明了选择理由，而非只写“同意 A/B”？

所有 92 个决议完成后，不要手工拼接最终 gold 文件。下一步应运行隔离包中的决议合并器，并用 `validate-gold` 对生成的单一 `gold_evidence_chains.json` 进行校验。
