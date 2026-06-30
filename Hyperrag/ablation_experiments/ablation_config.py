# -*- coding: utf-8 -*-
"""Configuration for HyperRAG v8.1 ablation experiments.

This file is self-contained: model/API settings, dataset paths, output paths,
and ablation switches are all configured here.
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

# ============================================================================
# Answer generation model
# ============================================================================
# Change these values to switch the model used to generate answers.
# LLM_MODEL = "Kimi-K2.5"
# LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
# LLM_API_KEY = "your-llm-api-key-here"

# # ============================================================================
# # Evaluation model
# # ============================================================================
# # Change these values to switch the model used to score answers.
# EVAL_MODEL = "Kimi-K2.5"
# EVAL_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
# EVAL_API_KEY = "your-eval-api-key-here"
LLM_BASE_URL = "https://api.openai-proxy.org/v1"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = "gpt-4o-mini"
EVAL_MODEL = "gpt-4o-mini"
EVAL_BASE_URL = "https://api.openai-proxy.org/v1"
EVAL_API_KEY = os.getenv("EVAL_API_KEY", LLM_API_KEY)
# ============================================================================
# Embedding model
# ============================================================================
EMB_MODEL = "BAAI/bge-m3"
EMB_BASE_URL = "https://api.siliconflow.cn/v1"
EMB_API_KEY = os.getenv("EMB_API_KEY", "")
EMB_DIM = 1024

# Dataset copied into this ablation folder.
DATA_NAME = "mechanical"
QUESTION_STAGE = 1
CACHES_DIR = BASE_DIR / "data" / "caches_v81"

# Ablation-local implementation and output paths.
RAG_V81_PATH = BASE_DIR / "rag_v81_ablation.py"
OUTPUT_DIR = BASE_DIR / "results"

# Ablation variants to run. Comment out entries here to run a subset.
RAG_METHODS = [
    "full_hyperrag_v81",
    "wo_attention_fusion_scores",
    "wo_query_cascade",
    "wo_topic_routing",
    "wo_entity_layer",
    "wo_multi_hop",
]

# Module switches. Each variant starts from the full model and flips one group.
ABLATION_VARIANTS = {
    "full_hyperrag_v81": {
        "label": "Full HyperRAG v8.1",
        "use_attention_fusion": True,
        "use_attention_scoring": True,
        "use_query_cascade": True,
        "use_topic_routing": True,
        "use_entity_layer": True,
        "enable_multi_hop": True,
    },
    "wo_attention_fusion_scores": {
        "label": "w/o Attention Fusion and Scores",
        "use_attention_fusion": False,
        "use_attention_scoring": False,
        "use_query_cascade": True,
        "use_topic_routing": True,
        "use_entity_layer": True,
        "enable_multi_hop": True,
    },
    "wo_query_cascade": {
        "label": "w/o Query Cascade",
        "use_attention_fusion": True,
        "use_attention_scoring": True,
        "use_query_cascade": False,
        "use_topic_routing": True,
        "use_entity_layer": True,
        "enable_multi_hop": True,
    },
    "wo_topic_routing": {
        "label": "w/o Topic Routing",
        "use_attention_fusion": True,
        "use_attention_scoring": True,
        "use_query_cascade": True,
        "use_topic_routing": False,
        "use_entity_layer": True,
        "enable_multi_hop": True,
    },
    "wo_entity_layer": {
        "label": "w/o Entity Layer",
        "use_attention_fusion": True,
        "use_attention_scoring": True,
        "use_query_cascade": True,
        "use_topic_routing": True,
        "use_entity_layer": False,
        "enable_multi_hop": True,
    },
    "wo_multi_hop": {
        "label": "w/o Multi-hop",
        "use_attention_fusion": True,
        "use_attention_scoring": True,
        "use_query_cascade": True,
        "use_topic_routing": True,
        "use_entity_layer": True,
        "enable_multi_hop": False,
    },
}

# Variants below only change query-time behavior, so they can reuse the same
# full HyperRAG v8.1 index/entity extraction cache.
SHARED_INDEX_VARIANTS = {
    "full_hyperrag_v81",
    "wo_query_cascade",
    "wo_topic_routing",
    "wo_multi_hop",
}

# Existing full-index cache generated with GPT-4o-mini entity extraction.
# If this file exists, the shared-index variants load it directly.
SHARED_INDEX_CACHE_PATH = BASE_DIR / "hyperrag_v81_gpt-4o-mini_6105367703bc2c1e.pkl"

# Kept for compatibility with copied evaluation code; ablation runner does not
# use these baselines unless you add them explicitly.
HYPERRAG_MAIN_MODE = "hyper"
HYPERRAG_MAIN_DIR = PROJECT_DIR / "Hyper-RAG-main"
LIGHTRAG_DIR = PROJECT_DIR / "LightRAG-main" / "LightRAG-main"
SIMPLE_RAG_PATH = PROJECT_DIR / "simple_rag.py"
