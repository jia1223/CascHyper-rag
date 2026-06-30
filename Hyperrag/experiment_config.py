# -*- coding: utf-8 -*-
"""
统一实验配置文件
用户只需修改 LLM_MODEL 即可切换大模型
"""

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# ============================================================================
# 生成回答的大模型（用户修改此处切换模型）
# ============================================================================
# LLM_MODEL = "qwen-max"
# LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# # LLM_API_KEY = "your-llm-api-key-here"
# LLM_API_KEY = "your-llm-api-key-here"
# # ============================================================================
# # 评判模型（固定 qwen-max）
# # ============================================================================
# EVAL_MODEL = "qwen-max"
# EVAL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# # EVAL_API_KEY = "your-eval-api-key-here"
# EVAL_API_KEY = "your-eval-api-key-here"
# LLM_MODEL = "deepseek-v3.2"
# LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
# LLM_API_KEY = "your-llm-api-key-here"
# EVAL_MODEL = "deepseek-v3.2"
# EVAL_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
# EVAL_API_KEY = "your-eval-api-key-here"
# LLM_MODEL = "glm-4.7"
# LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
# LLM_API_KEY = "your-llm-api-key-here"
# EVAL_MODEL = "glm-4.7"
# EVAL_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
# EVAL_API_KEY = "your-eval-api-key-here"
# LLM_MODEL = "Kimi-K2.5"
# LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
# LLM_API_KEY = "your-llm-api-key-here"
# EVAL_MODEL = "Kimi-K2.5"
# EVAL_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
# EVAL_API_KEY = "your-eval-api-key-here"
# LLM_BASE_URL = "https://api.openai-proxy.org/v1"
# LLM_API_KEY = "your-llm-api-key-here"
# LLM_MODEL = "qwen3.5-flash"
# EVAL_MODEL = "qwen3.5-flash"
# EVAL_BASE_URL = "https://api.openai-proxy.org/v1"
# EVAL_API_KEY = "your-eval-api-key-here"
# LLM_BASE_URL = "https://api.openai-proxy.org/v1"
# LLM_API_KEY = "your-llm-api-key-here"
# LLM_MODEL = "gpt-4o-mini"
# EVAL_MODEL = "gpt-4o-mini"
# EVAL_BASE_URL = "https://api.openai-proxy.org/v1"
# EVAL_API_KEY = "your-eval-api-key-here"
LLM_MODEL = "deepseek-v3.2"
LLM_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
EVAL_MODEL = "deepseek-v3.2"
EVAL_BASE_URL = "https://ark.cn-beijing.volces.com/api/coding/v3"
EVAL_API_KEY = os.getenv("EVAL_API_KEY", LLM_API_KEY)
# ============================================================================
# Embedding 配置
# ============================================================================
EMB_MODEL = "BAAI/bge-m3"
EMB_BASE_URL = "https://api.siliconflow.cn/v1"
EMB_API_KEY = os.getenv("EMB_API_KEY", "")
EMB_DIM = 1024

# ============================================================================
# 数据集配置
# ============================================================================
DATA_NAME = "hydraulics"                 # 数据集名称
QUESTION_STAGE = 1                      # 问题阶段 (1/2/3)
CACHES_DIR = BASE_DIR / "HyperRAG(8.1)" / "caches_v81"

# ============================================================================
# RAG 方法配置
# ============================================================================
# RAG_METHODS = [
#     "hyperrag_v81",       # HyperRAG v8.1（你自己设计的）
#     "lightrag",           # LightRAG
#     "simple_rag",         # 普通 RAG
#     "llm_only",           # 纯 LLM（无检索）
#     "hyperrag_main",      # Hyper-RAG-main
# ]
RAG_METHODS = [
    "hyperrag_v81",       # HyperRAG v8.1（你自己设计的）
    "lightrag",           # LightRAG
    "hyperrag_main",      # Hyper-RAG-main
]
# Hyper-RAG-main query mode. Valid values: "hyper", "hyper-lite", "graph", "naive", "llm".
HYPERRAG_MAIN_MODE = "hyper"

# RAG 实现路径
RAG_V81_PATH = BASE_DIR / "rag(8.1).py"
HYPERRAG_MAIN_DIR = BASE_DIR / "Hyper-RAG-main"
LIGHTRAG_DIR = BASE_DIR / "LightRAG-main" / "LightRAG-main"
SIMPLE_RAG_PATH = BASE_DIR / "simple_rag.py"

# ============================================================================
# 输出配置
# ============================================================================
OUTPUT_DIR = BASE_DIR / "experiment_results"
