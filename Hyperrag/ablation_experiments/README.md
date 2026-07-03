# HyperRAG v8.1 Ablation Experiments

This folder is isolated from the original project files. It contains:

- `rag_v81_ablation.py`: copied from `rag(8.1).py` and modified with module switches.
- `run_ablation_experiment.py`: copied from `run_experiment.py` and adapted to run ablation variants.
- `ablation_config.py`: single place to enable/disable variants and configure paths.
- `data/caches_v81/mechanical`: copied questions, references, and contexts for the current mechanical dataset.

Run from the project root or this folder:

```powershell
python ablation_experiments/run_ablation_experiment.py
```

To run only part of the ablation table, edit `RAG_METHODS` in `ablation_config.py`.

Current variants:

| Variant | Local Attention Fusion | Dynamic Multi-granularity Scoring | Query Cascade | Entity Layer | Multi-hop | Topic Routing |
| --- | --- | --- | --- | --- | --- | --- |
| `full_hyperrag_v81` | yes | yes | yes | yes | yes | yes |
| `wo_local_attention_fusion` | no | yes | yes | yes | yes | yes |
| `wo_dynamic_multigranularity_scoring` | yes | no | yes | yes | yes | yes |
| `wo_query_cascade` | yes | yes | no | yes | yes | yes |
| `wo_topic_routing` | yes | yes | yes | yes | yes | no |
| `wo_entity_layer` | yes | yes | yes | no | yes | yes |
| `wo_multi_hop` | yes | yes | yes | yes | no | yes |

Results are written to `ablation_experiments/results/<LLM_MODEL>/<DATA_NAME>/`.
