# -*- coding: utf-8 -*-
"""Configuration for retrieval-time and generation-token experiments.

This experiment is intentionally separate from the quality-scoring pipeline:
it measures one question per method, split into retrieval time and answer
generation time/token cost.
"""

from pathlib import Path
import sys


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import experiment_config as project_config

# Keep these defaults aligned with the project-level config because
# rag(8.1).py imports experiment_config internally. You can override them here;
# the runner patches experiment_config before loading HyperRAG v8.1.
LLM_MODEL = project_config.LLM_MODEL
LLM_BASE_URL = project_config.LLM_BASE_URL
LLM_API_KEY = project_config.LLM_API_KEY

EMB_MODEL = project_config.EMB_MODEL
EMB_BASE_URL = project_config.EMB_BASE_URL
EMB_API_KEY = project_config.EMB_API_KEY
EMB_DIM = project_config.EMB_DIM

# Dataset and one-question selection.
DATA_NAME = "mechanical"
QUESTION_STAGE = 1
QUESTION_INDEX = 0

# Put the experiment-local data under one of these supported layouts:
#   efficiency_experiments/data/mechanical/{questions,contexts}/
#   efficiency_experiments/data/caches_v81/mechanical/{questions,contexts}/
#   efficiency_experiments/mechanical/{questions,contexts}/
CACHES_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "results"
WORKING_CACHE_DIR = BASE_DIR / "cache"

# Plain RAG has no prepared framework cache. Keep this disabled when you want
# RAG to rebuild its chunk embeddings during each run.
SIMPLE_RAG_CACHE_ENABLED = False

# Implementations.
RAG_V81_PATH = PROJECT_DIR / "rag(8.1).py"
HYPERRAG_MAIN_DIR = PROJECT_DIR / "Hyper-RAG-main"
LIGHTRAG_DIR = PROJECT_DIR / "LightRAG-main" / "LightRAG-main"

# Methods compared in the efficiency experiment.
RAG_METHODS = [
    "simple_rag",
    "lightrag",
    "hyperrag_main",
    "hyperrag_v81",
]

METHOD_LABELS = {
    "simple_rag": "RAG",
    "lightrag": "LightRAG",
    "hyperrag_main": "HyperRAG",
    "hyperrag_v81": "HyperRAG v8.1",
}

# Retrieval settings kept close to run_experiment.py for fairness.
CHUNK_TOKEN_SIZE = 1200
CHUNK_OVERLAP_TOKEN_SIZE = 100
SIMPLE_RAG_TOP_K = 5
LIGHTRAG_MODE = "mix"
LIGHTRAG_CHUNK_TOP_K = 5
LIGHTRAG_TOP_K = 10
HYPERRAG_MAIN_MODE = "hyper"
HYPERRAG_MAIN_TOP_K = 10
HYPERRAG_V81_TOP_K_CHUNKS = 5
HYPERRAG_V81_TOP_K_SENTS = 10

# Generation is measured with a single shared prompt for all methods so that
# token cost reflects retrieved context size plus the answer itself.
GENERATION_SYSTEM_PROMPT = (
    "You are a careful domain question-answering assistant. Answer only from "
    "the provided retrieved context. If the context is insufficient, say so."
)
