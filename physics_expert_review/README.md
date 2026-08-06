# Physics 专家审核实验包

这是一个与现有 RAG 代码隔离的实验包，用于比较 **CascHyper-RAG** 与 **Hyper-RAG** 在 Physics 数据集上的专家答案质量评分。

它不运行 RAG、不修改现有结果文件，也不评估检索证据链。它只完成以下流程：

```text
既有两种方法的答案结果
→ 固定随机抽样与匿名 A/B 评分表
→ 专家独立打分
→ 解盲后计算均分、95% paired-bootstrap CI、胜平负和 ICC
```

## 目录说明

```text
physics_expert_review/
├── prepare_review.py          # 生成固定样本、参考答案模板和匿名评分表
├── analyze_reviews.py         # 读取完成的评分表并生成统计结果
├── run_prepare_physics.ps1    # 针对当前 Physics 结果文件的准备命令
├── 专家评分指南.md             # 发给专家的评分与校准说明
├── tests/                     # 命令行端到端测试
└── runs/                      # 本地实验运行产物；默认不纳入版本控制
```

## 当前输入

`run_prepare_physics.ps1` 固定使用以下已有结果：

- CascHyper-RAG：`Hyperrag/experiment_results/deepseek-v4-flash/physics/hyperrag_v81_<stage>_stage_result.json`
- Hyper-RAG：`Hyperrag/experiment_results/deepseek-v4-flash/physics/hyperrag_main_<stage>_stage_result.json`

三个阶段共 150 个配对问题。脚本以固定随机种子抽取 60 个正式题和 5 个校准题。

## 运行步骤

### 1. 固定问题样本并生成参考答案模板

在仓库根目录运行：

```powershell
powershell -ExecutionPolicy Bypass -File .\physics_expert_review\run_prepare_physics.ps1
```

首次运行会在 `runs/physics_expert_review_seed_20260806/` 创建 `reference_answers_template.json` 与 `run_metadata_template.json`，然后以退出码 2 正常停止。这是设计使然：当前结果 JSON 只保存问题和系统答案，没有可直接用于专家审核的标准答案，也不含足以审计的完整冻结配置。

请由研究者根据原始 Physics 语料或既有金标准，为模板内每一题填写 `reference_answer`。参考答案应在发给专家前冻结。

同时填写 `run_metadata_template.json`，记录两种方法的代码版本、生成模型、提示词版本、解码参数、上下文 token 预算和 embedding 模型。正式生成评分表时，脚本还会保存六个输入结果文件的 SHA-256 哈希、65 道抽样题、随机种子和创建日期。

### 2. 生成匿名专家评分表

填完两个模板后，以它们作为 `--references` 和 `--run-metadata` 参数重新运行准备命令。可直接复制 `run_prepare_physics.ps1` 中的命令并在末尾加入：

```text
--references .\physics_expert_review\runs\physics_expert_review_seed_20260806\reference_answers_template.json --run-metadata .\physics_expert_review\runs\physics_expert_review_seed_20260806\run_metadata_template.json --overwrite
```

成功后生成：

- `materials/expert_01.csv` 至 `materials/expert_03.csv`：正式评分表；
- `materials/calibration_expert_*.csv`：仅用于校准的 5 题；
- `private/unblinding_key.csv`：A/B 到方法的映射，必须在评分锁定前隔离保存；
- `private/sample_manifest.json`：完整抽样题、输入文件哈希和冻结运行元数据。

表单内每题的两份答案会按随机的 `A/B` 或 `B/A` 顺序展示。只能将 `materials/` 中的文件发给专家；绝不能发送 `private/`。

### 3. 回收评分表

专家完成评分后，将每位专家的正式表保存到同一目录，例如：

```text
runs/physics_expert_review_seed_20260806/completed_forms/expert_01.csv
runs/physics_expert_review_seed_20260806/completed_forms/expert_02.csv
runs/physics_expert_review_seed_20260806/completed_forms/expert_03.csv
```

每位专家必须对每道正式题填写：

- System A 与 System B 的 Correctness、Completeness、Clarity（均为 1--5 的整数）；
- `overall_preference`：`A`、`B` 或 `Tie`。

### 4. 解盲并生成统计结果

只有在所有评分表锁定后，运行：

```powershell
$py = 'C:\Users\kunkun\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py .\physics_expert_review\analyze_reviews.py `
  --forms-dir .\physics_expert_review\runs\physics_expert_review_seed_20260806\completed_forms `
  --unblinding-key .\physics_expert_review\runs\physics_expert_review_seed_20260806\private\unblinding_key.csv `
  --output-dir .\physics_expert_review\runs\physics_expert_review_seed_20260806\analysis
```

输出包括：

- `analysis/report.md`：可直接用于论文结果整理的汇总表；
- `analysis/summary.json`：完整统计数值；
- `analysis/per_question_scores.csv`：逐题平均分和方法差值。

## 统计口径

- 每题、每方法先对所有专家评分取平均；
- 以逐题 `CascHyper-RAG - Hyper-RAG` 差值进行 10,000 次 paired bootstrap；
- 报告 Correctness、Completeness、Clarity 的均分、均值差和 95% CI；
- Overall preference 按全部专家的成对投票汇总为 CascHyper-RAG 胜 / 平 / Hyper-RAG 胜；
- 使用 two-way random-effects, absolute-agreement ICC(A,1) 评估专家评分一致性。

## 验证

无需额外安装 Python 包。运行：

```powershell
$py = 'C:\Users\kunkun\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
& $py -m unittest discover -s .\physics_expert_review\tests -v
```
