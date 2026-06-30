# -*- coding: utf-8 -*-
"""
统一实验配置文件
用户只需修改 LLM_MODEL 即可切换大模型
"""

# ============================================================================
# 生成回答的大模型（用户修改此处切换模型）
# ============================================================================
import os

LLM_MODEL = "qwen-max"
LLM_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

# ============================================================================
# 评判模型（固定 qwen-max）
# ============================================================================
EVAL_MODEL = "qwen-max"
EVAL_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EVAL_API_KEY = os.getenv("EVAL_API_KEY", LLM_API_KEY)

# ============================================================================
# Embedding 配置
# ============================================================================
EMB_MODEL = "text-embedding-3-small"
EMB_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
EMB_API_KEY = os.getenv("EMB_API_KEY", "")
EMB_DIM = 1536

# ============================================================================
# 数据集配置
# ============================================================================
DATA_NAME = "pathology"                 # 数据集名称
QUESTION_STAGE = 1                      # 问题阶段 (1/2/3)
CACHES_DIR = r"e:\HyperGraphRAG-main\Hyperrag\HyperRAG(8.1)\caches_v81"

# ============================================================================
# RAG 方法配置
# ============================================================================
# 要运行的 RAG 方法列表（可注释掉不需要的）
RAG_METHODS = [
    "hyperrag_v81",       # HyperRAG v8.1（你自己设计的）
    "hyperrag_main",      # Hyper-RAG-main
    "lightrag",           # LightRAG
    "simple_rag",         # 普通 RAG
    "llm_only",           # 纯 LLM（无检索）
]

# RAG 实现路径
RAG_V81_PATH = r"e:\HyperGraphRAG-main\Hyperrag\rag(8.1).py"
HYPERRAG_MAIN_DIR = r"e:\HyperGraphRAG-main\Hyperrag\Hyper-RAG-main"
LIGHTRAG_DIR = r"e:\HyperGraphRAG-main\Hyperrag\LightRAG-main\LightRAG-main"
SIMPLE_RAG_PATH = r"e:\HyperGraphRAG-main\Hyperrag\simple_rag.py"

# ============================================================================
# 输出配置
# ============================================================================
OUTPUT_DIR = r"e:\HyperGraphRAG-main\Hyperrag\experiment_results"
