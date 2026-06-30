"""
Step 3: 答案生成
使用 HyperRAG_Attention_v81 引擎对 Step 2 生成的问题进行 search → verify → generate。
"""
import sys
import json
import asyncio
import argparse
import importlib.util
from pathlib import Path
from tqdm import tqdm

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


async def process_single_query(engine, query_text):
    """对单个问题执行完整的 search → verify → generate 流程"""
    try:
        # 检索
        results = await engine.search(query_text, top_k_chunks=10, top_k_sents=10)
        # 校验
        results = await engine.verify_results(query_text, results)
        # 生成
        answer = await engine.generate_answer(query_text, results)
        return {"query": query_text, "result": answer}, None
    except Exception as e:
        print(f"\n  [Error] Query failed: {e}")
        return None, {"query": query_text, "error": str(e)}


async def run_all_queries(data_name: str, question_stage: int):
    """加载引擎和问题，逐一处理"""
    caches_root = Path(CACHES_DIR)

    # 加载问题
    question_file = caches_root / data_name / "questions" / f"{question_stage}_stage.json"
    if not question_file.exists():
        print(f"[Error] Question file not found: {question_file}")
        print("  请先运行 step_2_extract_question.py")
        return

    with open(question_file, "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"[Load] {len(queries)} questions loaded from {question_file.name}")

    # 加载引擎索引
    context_file = caches_root / data_name / "contexts" / f"{data_name}_unique_contexts.json"
    if not context_file.exists():
        print(f"[Error] Context file not found. Please run step_0 and step_1 first.")
        return

    with open(context_file, "r", encoding="utf-8") as f:
        unique_contexts = json.load(f)
    raw_text = "\n\n".join(unique_contexts)

    # 设置缓存目录
    engine_cache_dir = str(caches_root / data_name / "engine_cache")
    rag_v81.CACHE_DIR = engine_cache_dir

    engine = HyperRAG_Attention_v81()
    cache_path = get_cache_path(raw_text, prefix=f"hyperrag_v81_{data_name}")

    if not load_engine_state(engine, cache_path):
        print("[Error] No cached engine index found. Please run step_1 first.")
        return

    # 输出目录
    out_dir = caches_root / data_name / "response"
    out_dir.mkdir(parents=True, exist_ok=True)

    result_file = out_dir / f"v81_{question_stage}_stage_result.json"
    error_file = out_dir / f"v81_{question_stage}_stage_errors.json"

    results_list = []
    errors_list = []

    for query_text in tqdm(queries, desc="Processing queries", unit="query"):
        result, error = await process_single_query(engine, query_text)
        if result:
            results_list.append(result)
        elif error:
            errors_list.append(error)

    # 保存结果
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)

    if errors_list:
        with open(error_file, "w", encoding="utf-8") as f:
            json.dump(errors_list, f, ensure_ascii=False, indent=4)

    print(f"\n[Done] {len(results_list)} answers generated, {len(errors_list)} errors.")
    print(f"  Results: {result_file}")
    if errors_list:
        print(f"  Errors:  {error_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: Generate answers using HyperRAG v8.1 engine")
    parser.add_argument("-d", "--data_name", type=str, default="mix")
    parser.add_argument("-s", "--stage", type=int, default=2, choices=[1, 2, 3])
    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"Step 3: Generating answers for '{args.data_name}' ({args.stage}-stage)")
    print(f"{'='*60}")

    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(run_all_queries(args.data_name, args.stage))
