"""
Step 1: 构建知识索引
加载去重 context，使用 HyperRAG_Attention_v81 引擎构建级联索引并缓存。
"""
import sys
import json
import asyncio
import argparse
import importlib.util
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from my_config import RAG_FILE_PATH, HYPERRAG_LIB_DIR, CACHES_DIR

# --- 动态导入 rag(8.1).py ---
sys.path.insert(0, HYPERRAG_LIB_DIR)

spec = importlib.util.spec_from_file_location("rag_v81", RAG_FILE_PATH)
rag_v81 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rag_v81)

HyperRAG_Attention_v81 = rag_v81.HyperRAG_Attention_v81
save_engine_state = rag_v81.save_engine_state
load_engine_state = rag_v81.load_engine_state
get_cache_path = rag_v81.get_cache_path


async def build_index_for_dataset(data_name: str):
    """为指定数据集构建索引"""
    caches_root = Path(CACHES_DIR)
    context_file = caches_root / data_name / "contexts" / f"{data_name}_unique_contexts.json"

    if not context_file.exists():
        print(f"[Error] Context file not found: {context_file}")
        print("  请先运行 step_0_preprocess.py")
        return

    print(f"[Load] Reading contexts from {context_file}")
    with open(context_file, "r", encoding="utf-8") as f:
        unique_contexts = json.load(f)

    # 将所有 context 拼接为一个长文本
    raw_text = "\n\n".join(unique_contexts)
    print(f"[Data] {len(unique_contexts)} contexts, total {len(raw_text)} characters")

    # 设置缓存目录
    engine_cache_dir = str(caches_root / data_name / "engine_cache")
    rag_v81.CACHE_DIR = engine_cache_dir

    # 创建引擎
    engine = HyperRAG_Attention_v81()

    # 检查缓存
    cache_path = get_cache_path(raw_text, prefix=f"hyperrag_v81_{data_name}")
    if load_engine_state(engine, cache_path):
        print("[Cache] Index loaded from cache, skipping build.")
        return

    print("[Build] Starting index construction...")
    await engine.build_index(
        raw_text,
        max_token_size=400,
        overlap_token_size=50,
        overlap_threshold=0.6
    )

    save_engine_state(engine, cache_path)
    print(f"[Done] Index built and cached for '{data_name}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Build knowledge index")
    parser.add_argument("-d", "--data_name", type=str, default="mix")
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Step 1: Building index for '{args.data_name}'")
    print(f"{'='*60}")

    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(build_index_for_dataset(args.data_name))
