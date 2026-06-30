# Query Scoring Comparison

This folder contains a focused comparison for the two query-scoring formulas.
It does not rebuild the HyperRAG index. It uses the cached index already placed
in this folder:

```text
duibi/hyperrag_v81_gpt-4o-mini_6105367703bc2c1e.pkl
```

## Compared Methods

`shared_exp_query`

```text
s_c = q_exp dot h_c(final)
s_s = q_exp dot h_s(final)
s_e = entity MaxSim
```

`level_specific_query`

```text
s_c = q_c dot h_c(final)
s_s = q_s dot h_s(final)
s_e = entity MaxSim
```

Both methods use the same query expansion, entity extraction, coarse chunk
retrieval, answer prompt, and six-metric LLM scoring rubric.

## Run

Set the API environment variables first:

```powershell
$env:OPENAI_API_KEY="your-4o-mini-key"
$env:OPENAI_BASE_URL="https://api.openai.com/v1"
$env:EMB_API_KEY="your-embedding-key"
$env:EMB_BASE_URL="https://api.siliconflow.cn/v1"
$env:EMB_MODEL="BAAI/bge-m3"
```

Then run:

```powershell
D:\miniconda\envs\rag\python.exe duibi\query_scoring_compare.py --stage 1
```

For a quick one-question check:

```powershell
D:\miniconda\envs\rag\python.exe duibi\query_scoring_compare.py --stage 1 --limit 1
```

## Outputs

Results are written under:

```text
duibi/results/gpt-4o-mini/mechanical/
```

Main files:

```text
shared_exp_query_1_stage_result.json
shared_exp_query_1_stage_retrieval.json
shared_exp_query_1_stage_scoring.json
level_specific_query_1_stage_result.json
level_specific_query_1_stage_retrieval.json
level_specific_query_1_stage_scoring.json
summary_query_scoring_1_stage.json
```

The summary file reports the average score for:

```text
Comprehensiveness, Diversity, Empowerment, Logical, Readability, Relevance, Average
```
