# Retrieval Time and Token Cost Experiments

This folder is independent from `ablation_experiments`. It compares:

- `simple_rag` as plain RAG
- `lightrag` as LightRAG
- `hyperrag_main` as HyperRAG baseline
- `hyperrag_v81` as our HyperRAG v8.1

The runner measures one selected question:

- retrieval time: only the time to retrieve/build context for that question
- generation time: only the final answer-generation API call
- generation prompt tokens
- answer/completion tokens
- total generation tokens

It does not include scoring-model tokens, and it does not include offline index
construction time in the reported retrieval/generation metrics.

## Run

```powershell
python efficiency_experiments\run_efficiency_experiment.py
```

Edit `efficiency_config.py` to change `DATA_NAME`, `QUESTION_STAGE`,
`QUESTION_INDEX`, model settings, or method list.

The experiment reads data from the local efficiency folder. Supported layouts:

```text
efficiency_experiments/data/mechanical/{questions,contexts}/
efficiency_experiments/data/caches_v81/mechanical/{questions,contexts}/
efficiency_experiments/mechanical/{questions,contexts}/
```

Outputs are written to:

```text
efficiency_experiments/results/<model>/<dataset>/
```

The main files are:

- `efficiency_detail_<stage>_q<index>.json`
- `efficiency_summary_<stage>_q<index>.csv`

## Cache Isolation

Prepared framework caches are used directly from:

```text
efficiency_experiments/cache/<method>/<model>/<dataset>/<stage>/
```

For this run, the cache model folder is `deepseek-v3.2`. Plain RAG has no
prepared cache and rebuilds its temporary chunk embeddings during the run.
HyperRAG v8.1's cache is read from `efficiency_experiments/cache/hyperrag_v81`.
