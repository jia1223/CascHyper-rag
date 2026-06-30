# -*- coding: utf-8 -*-
"""One-question retrieval-time and generation-token comparison.

The experiment separates each method into:
1. retrieval/context construction time
2. final answer generation time and token usage

The final answer is generated with one shared prompt for all methods, so token
comparison is driven by each retriever's returned context rather than by
different answer-generation templates hidden inside each framework.
"""

import asyncio
import csv
import hashlib
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from openai import OpenAI

import efficiency_config as cfg


STAGE_CACHE_NAME = f"{cfg.QUESTION_STAGE}_stage"
CACHE_STATUS_LOG: list[dict[str, Any]] = []


def safe_cache_name(value: Any) -> str:
    return re.sub(r"[^\w\-.]", "_", str(value))


def cache_model_name() -> str:
    return safe_cache_name(getattr(cfg, "CACHE_MODEL_NAME", cfg.LLM_MODEL))


def method_cache_root(method_name: str) -> Path:
    return Path(cfg.WORKING_CACHE_DIR) / method_name


def method_working_dir(method_name: str) -> Path:
    root = method_cache_root(method_name)
    if method_name == "hyperrag_v81":
        return root
    return (
        root
        / cache_model_name()
        / cfg.DATA_NAME
        / STAGE_CACHE_NAME
    )


def inspect_efficiency_caches() -> list[dict[str, Any]]:
    log: list[dict[str, Any]] = []
    for method in cfg.RAG_METHODS:
        if method == "simple_rag":
            log.append(
                {
                    "method": method,
                    "status": "no_prepared_cache",
                    "note": "Plain RAG builds its own temporary embeddings for this run.",
                }
            )
            continue
        cache_root = method_cache_root(method)
        working_dir = method_working_dir(method)
        files = [p for p in working_dir.rglob("*") if p.is_file()] if working_dir.exists() else []
        log.append(
            {
                "method": method,
                "status": "found" if files else "missing",
                "cache_root": str(cache_root),
                "working_dir": str(working_dir),
                "file_count": len(files),
            }
        )
    return log


def make_openai_client(api_key: str, base_url: str) -> OpenAI:
    return OpenAI(api_key=api_key, base_url=base_url, max_retries=5, timeout=120)


def count_tokens(text: str) -> int:
    try:
        import tiktoken

        try:
            enc = tiktoken.encoding_for_model(str(cfg.LLM_MODEL))
        except KeyError:
            enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text or ""))
    except Exception:
        return max(1, len(text or "") // 4)


def load_questions_and_contexts() -> tuple[list[str], list[str]]:
    candidate_roots = [
        Path(cfg.CACHES_DIR),
        Path(cfg.CACHES_DIR) / "caches_v81",
        Path(cfg.BASE_DIR) / "data",
        Path(cfg.BASE_DIR) / "data" / "caches_v81",
        Path(cfg.BASE_DIR),
    ]
    seen_roots = set()
    checked_pairs: list[tuple[Path, Path]] = []
    questions_file = None
    contexts_file = None
    for root in candidate_roots:
        root = Path(root)
        root_key = str(root.resolve()) if root.exists() else str(root)
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        q_file = root / cfg.DATA_NAME / "questions" / f"{cfg.QUESTION_STAGE}_stage.json"
        c_file = root / cfg.DATA_NAME / "contexts" / f"{cfg.DATA_NAME}_unique_contexts.json"
        checked_pairs.append((q_file, c_file))
        if q_file.exists() and c_file.exists():
            questions_file = q_file
            contexts_file = c_file
            break

    if questions_file is None or contexts_file is None:
        checked = "\n".join(f"  - {q}\n  - {c}" for q, c in checked_pairs)
        raise FileNotFoundError(
            "Experiment-local mechanical data not found. Expected questions and "
            "contexts under efficiency_experiments/data or efficiency_experiments/mechanical.\n"
            f"Checked:\n{checked}"
        )

    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(contexts_file, "r", encoding="utf-8") as f:
        contexts = json.load(f)

    if not 0 <= cfg.QUESTION_INDEX < len(questions):
        raise IndexError(
            f"QUESTION_INDEX={cfg.QUESTION_INDEX} is out of range for "
            f"{len(questions)} questions"
        )
    return questions, contexts


def build_generation_prompt(question: str, context: str) -> str:
    return f"""Retrieved context:
{context or "No retrieved context."}

Question:
{question}

Answer:"""


def generate_answer(question: str, context: str) -> tuple[str, dict[str, int]]:
    prompt = build_generation_prompt(question, context)
    client = make_openai_client(cfg.LLM_API_KEY, cfg.LLM_BASE_URL)
    response = client.chat.completions.create(
        model=cfg.LLM_MODEL,
        messages=[
            {"role": "system", "content": cfg.GENERATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    answer = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)

    if usage is not None:
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        answer_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(getattr(usage, "total_tokens", 0) or 0)
    else:
        prompt_tokens = count_tokens(cfg.GENERATION_SYSTEM_PROMPT) + count_tokens(prompt)
        answer_tokens = count_tokens(answer)
        total_tokens = prompt_tokens + answer_tokens

    return answer, {
        "generation_prompt_tokens": prompt_tokens,
        "answer_tokens": answer_tokens,
        "generation_total_tokens": total_tokens,
    }


async def global_llm_func(prompt: str, system_prompt: str | None = None, **kwargs):
    history_messages = kwargs.pop("history_messages", []) or []
    hashing_kv = kwargs.pop("hashing_kv", None)
    model = kwargs.pop("model", cfg.LLM_MODEL)
    base_url = kwargs.pop("base_url", cfg.LLM_BASE_URL)
    api_key = kwargs.pop("api_key", cfg.LLM_API_KEY)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    request_kwargs = {}
    for key in (
        "temperature",
        "top_p",
        "max_tokens",
        "presence_penalty",
        "frequency_penalty",
        "stop",
    ):
        if key in kwargs and kwargs[key] is not None:
            request_kwargs[key] = kwargs[key]

    cache_key = None
    if hashing_kv is not None:
        cache_key = hashlib.md5(str((model, messages, request_kwargs)).encode()).hexdigest()
        cached = await hashing_kv.get_by_id(cache_key)
        if cached is not None:
            return cached["return"]

    client = make_openai_client(api_key, base_url)
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        **request_kwargs,
    )
    result = response.choices[0].message.content or ""

    if hashing_kv is not None:
        await hashing_kv.upsert({cache_key: {"return": result, "model": model}})
    return result


async def global_emb_func_list(texts: list[str]):
    client = make_openai_client(cfg.EMB_API_KEY, cfg.EMB_BASE_URL)
    resp = client.embeddings.create(model=cfg.EMB_MODEL, input=texts)
    return [item.embedding for item in resp.data]


async def global_emb_func_ndarray(texts: list[str]):
    return np.array(await global_emb_func_list(texts))


class BaseEfficiencyAdapter:
    name = "base"

    def retrieve_context(self, question: str) -> tuple[str, dict[str, Any]]:
        raise NotImplementedError

    def cleanup(self):
        pass


class SimpleRAGEfficiencyAdapter(BaseEfficiencyAdapter):
    name = "simple_rag"

    def __init__(self, contexts: list[str]):
        import tiktoken

        self.emb_client = make_openai_client(cfg.EMB_API_KEY, cfg.EMB_BASE_URL)
        try:
            self.encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

        self.cache_dir = (
            Path(cfg.WORKING_CACHE_DIR)
            / "simple_rag"
            / safe_cache_name(cfg.EMB_MODEL)
            / cfg.DATA_NAME
            / STAGE_CACHE_NAME
        )
        if self._load_cache():
            return

        self.chunks: list[str] = []
        for ctx in contexts:
            tokens = self.encoder.encode(ctx)
            if len(tokens) <= cfg.CHUNK_TOKEN_SIZE:
                self.chunks.append(ctx)
                continue
            start = 0
            while start < len(tokens):
                end = min(start + cfg.CHUNK_TOKEN_SIZE, len(tokens))
                self.chunks.append(self.encoder.decode(tokens[start:end]))
                start += cfg.CHUNK_TOKEN_SIZE - cfg.CHUNK_OVERLAP_TOKEN_SIZE

        all_embeddings = []
        for i in range(0, len(self.chunks), 20):
            batch = self.chunks[i : i + 20]
            resp = self.emb_client.embeddings.create(model=cfg.EMB_MODEL, input=batch)
            all_embeddings.extend([item.embedding for item in resp.data])

        self.embeddings = np.array(all_embeddings, dtype=float)
        if len(self.embeddings):
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.embeddings = self.embeddings / norms
        self._save_cache()

    def _load_cache(self) -> bool:
        if not getattr(cfg, "SIMPLE_RAG_CACHE_ENABLED", True):
            return False
        chunks_file = self.cache_dir / "chunks.json"
        embeddings_file = self.cache_dir / "embeddings.npy"
        if not chunks_file.exists() or not embeddings_file.exists():
            return False
        try:
            with open(chunks_file, "r", encoding="utf-8") as f:
                chunks = json.load(f)
            embeddings = np.load(embeddings_file)
        except Exception:
            return False
        if not isinstance(chunks, list) or len(chunks) != len(embeddings):
            return False
        self.chunks = [str(item) for item in chunks]
        self.embeddings = np.array(embeddings, dtype=float)
        return True

    def _save_cache(self):
        if not getattr(cfg, "SIMPLE_RAG_CACHE_ENABLED", True):
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        with open(self.cache_dir / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False)
        np.save(self.cache_dir / "embeddings.npy", self.embeddings)

    def _embed_query(self, question: str) -> np.ndarray:
        resp = self.emb_client.embeddings.create(model=cfg.EMB_MODEL, input=[question])
        q_vec = np.array(resp.data[0].embedding, dtype=float)
        norm = np.linalg.norm(q_vec)
        return q_vec / norm if norm > 0 else q_vec

    def retrieve_context(self, question: str) -> tuple[str, dict[str, Any]]:
        if not len(self.chunks):
            return "", {"retrieved_items": 0}

        q_vec = self._embed_query(question)
        scores = self.embeddings @ q_vec
        top_k = min(cfg.SIMPLE_RAG_TOP_K, len(self.chunks))
        indices = np.argsort(scores)[-top_k:][::-1]
        parts = [
            f"[Document Chunk {rank}] (score={scores[idx]:.4f})\n{self.chunks[idx]}"
            for rank, idx in enumerate(indices, start=1)
        ]
        return "\n\n".join(parts), {
            "retrieved_items": int(top_k),
            "top_scores": [float(scores[idx]) for idx in indices],
        }


class HyperRAGV81EfficiencyAdapter(BaseEfficiencyAdapter):
    name = "hyperrag_v81"

    def __init__(self, contexts: list[str]):
        self.loop = asyncio.new_event_loop()
        self.contexts = contexts
        self.engine = None
        self._init_engine()

    def _patch_project_config(self):
        import experiment_config as project_config

        for key in (
            "LLM_MODEL",
            "LLM_BASE_URL",
            "LLM_API_KEY",
            "EMB_MODEL",
            "EMB_BASE_URL",
            "EMB_API_KEY",
            "EMB_DIM",
        ):
            if hasattr(cfg, key):
                setattr(project_config, key, getattr(cfg, key))

    def _init_engine(self):
        self._patch_project_config()
        for module_name in list(sys.modules):
            if module_name == "hyperrag" or module_name.startswith("hyperrag."):
                del sys.modules[module_name]

        hyperrag_pkg = str(cfg.PROJECT_DIR / "Hyper-RAG")
        if hyperrag_pkg in sys.path:
            sys.path.remove(hyperrag_pkg)
        sys.path.insert(0, hyperrag_pkg)

        spec = importlib.util.spec_from_file_location("rag_v81_efficiency", cfg.RAG_V81_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot load HyperRAG v8.1 from {cfg.RAG_V81_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules["rag_v81_efficiency"] = module
        spec.loader.exec_module(module)
        v81_cache_dir = method_working_dir("hyperrag_v81")
        v81_cache_dir.mkdir(parents=True, exist_ok=True)
        if hasattr(module, "CACHE_DIR"):
            module.CACHE_DIR = str(v81_cache_dir)

        rag_class = getattr(module, "HyperRAG_Attention_v81", None)
        if rag_class is None:
            raise RuntimeError("HyperRAG_Attention_v81 class not found")

        self.engine = rag_class()
        raw_text = "\n\n".join(self.contexts)
        asyncio.set_event_loop(self.loop)

        get_cache_path = getattr(module, "get_cache_path", None)
        load_engine_state = getattr(module, "load_engine_state", None)
        save_engine_state = getattr(module, "save_engine_state", None)
        cache_path = None
        cache_loaded = False
        if get_cache_path and load_engine_state:
            cache_path = get_cache_path(
                raw_text,
                prefix=f"hyperrag_v81_{STAGE_CACHE_NAME}",
                model=cache_model_name(),
            )
            cache_loaded = load_engine_state(self.engine, cache_path)

        if not cache_loaded:
            self.loop.run_until_complete(
                self.engine.build_index(
                    raw_text,
                    max_token_size=cfg.CHUNK_TOKEN_SIZE,
                    overlap_token_size=cfg.CHUNK_OVERLAP_TOKEN_SIZE,
                )
            )
            if save_engine_state and cache_path:
                save_engine_state(self.engine, cache_path)

    def retrieve_context(self, question: str) -> tuple[str, dict[str, Any]]:
        asyncio.set_event_loop(self.loop)
        results = self.loop.run_until_complete(
            self.engine.search(
                question,
                top_k_chunks=cfg.HYPERRAG_V81_TOP_K_CHUNKS,
                top_k_sents=cfg.HYPERRAG_V81_TOP_K_SENTS,
            )
        )
        results = self.loop.run_until_complete(self.engine.verify_results(question, results))

        parts = []
        for i, item in enumerate(results.get("top_chunks", [])[: cfg.HYPERRAG_V81_TOP_K_CHUNKS], 1):
            parts.append(
                f"[Standard RAG Evidence {i}] (score={item.get('score', 0):.4f})\n"
                f"{item.get('text', '')}"
            )
        for i, item in enumerate(results.get("hop1", []), 1):
            parts.append(f"[Graph Evidence {i}]\n{item.get('expanded_text', item.get('text', ''))}")
        for i, item in enumerate(results.get("hop2", []), 1):
            parts.append(
                f"[Multi-hop Graph Evidence {i}] via {item.get('via', '')}\n"
                f"{item.get('expanded_text', item.get('text', ''))}"
            )

        return "\n\n".join(parts), {
            "retrieved_items": len(parts),
            "max_score": float(results.get("max_score", 0.0)),
        }

    def cleanup(self):
        if self.loop and not self.loop.is_closed():
            asyncio.set_event_loop(None)
            self.loop.close()


class HyperRAGMainEfficiencyAdapter(BaseEfficiencyAdapter):
    name = "hyperrag_main"

    def __init__(self, contexts: list[str]):
        self.loop = asyncio.new_event_loop()
        self.contexts = contexts
        self.rag = None
        self._init_engine()

    def _init_engine(self):
        for module_name in list(sys.modules):
            if module_name == "hyperrag" or module_name.startswith("hyperrag."):
                del sys.modules[module_name]
        main_dir = str(cfg.HYPERRAG_MAIN_DIR)
        if main_dir in sys.path:
            sys.path.remove(main_dir)
        sys.path.insert(0, main_dir)

        from hyperrag import HyperRAG
        from hyperrag.utils import EmbeddingFunc

        working_dir = method_working_dir("hyperrag_main")
        working_dir.mkdir(parents=True, exist_ok=True)
        self.rag = HyperRAG(
            working_dir=str(working_dir),
            llm_model_func=global_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=cfg.EMB_DIM,
                max_token_size=8192,
                func=global_emb_func_list,
            ),
            chunk_token_size=cfg.CHUNK_TOKEN_SIZE,
            chunk_overlap_token_size=cfg.CHUNK_OVERLAP_TOKEN_SIZE,
        )
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.rag.ainsert(self.contexts))

    def retrieve_context(self, question: str) -> tuple[str, dict[str, Any]]:
        asyncio.set_event_loop(self.loop)
        from hyperrag.base import QueryParam

        param = QueryParam(
            mode=cfg.HYPERRAG_MAIN_MODE,
            only_need_context=True,
            top_k=cfg.HYPERRAG_MAIN_TOP_K,
            max_token_for_text_unit=cfg.CHUNK_TOKEN_SIZE,
        )
        context = self.loop.run_until_complete(self.rag.aquery(question, param))
        if not isinstance(context, str):
            context = json.dumps(context, ensure_ascii=False)
        return context, {"retrieved_items": context.count("\n") + 1 if context else 0}

    def cleanup(self):
        if self.loop and not self.loop.is_closed():
            asyncio.set_event_loop(None)
            self.loop.close()


class LightRAGEfficiencyAdapter(BaseEfficiencyAdapter):
    name = "lightrag"

    def __init__(self, contexts: list[str]):
        self.loop = asyncio.new_event_loop()
        self.contexts = contexts
        self.rag = None
        self._init_engine()

    def _init_engine(self):
        light_dir = str(cfg.LIGHTRAG_DIR)
        if light_dir in sys.path:
            sys.path.remove(light_dir)
        sys.path.insert(0, light_dir)

        from lightrag import LightRAG
        from lightrag.utils import EmbeddingFunc

        working_dir = method_working_dir("lightrag")
        working_dir.mkdir(parents=True, exist_ok=True)

        self.rag = LightRAG(
            working_dir=str(working_dir),
            llm_model_func=global_llm_func,
            embedding_func=EmbeddingFunc(
                embedding_dim=cfg.EMB_DIM,
                max_token_size=8192,
                func=global_emb_func_ndarray,
            ),
            chunk_token_size=cfg.CHUNK_TOKEN_SIZE,
            chunk_overlap_token_size=cfg.CHUNK_OVERLAP_TOKEN_SIZE,
        )
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.rag.initialize_storages())
        self.loop.run_until_complete(self.rag.ainsert(self.contexts))

    def retrieve_context(self, question: str) -> tuple[str, dict[str, Any]]:
        asyncio.set_event_loop(self.loop)
        from lightrag import QueryParam

        param = QueryParam(
            mode=cfg.LIGHTRAG_MODE,
            only_need_context=True,
            chunk_top_k=cfg.LIGHTRAG_CHUNK_TOP_K,
            top_k=cfg.LIGHTRAG_TOP_K,
        )
        context = self.loop.run_until_complete(self.rag.aquery(question, param))
        if not isinstance(context, str):
            context = json.dumps(context, ensure_ascii=False)
        return context, {"retrieved_items": context.count("\n") + 1 if context else 0}

    def cleanup(self):
        if self.rag is not None and self.loop and not self.loop.is_closed():
            asyncio.set_event_loop(self.loop)
            finalize = getattr(self.rag, "finalize_storages", None)
            if callable(finalize):
                result = finalize()
                if asyncio.iscoroutine(result):
                    self.loop.run_until_complete(result)
        if self.loop and not self.loop.is_closed():
            asyncio.set_event_loop(None)
            self.loop.close()


def create_adapter(method_name: str, contexts: list[str]) -> BaseEfficiencyAdapter:
    adapters = {
        "simple_rag": SimpleRAGEfficiencyAdapter,
        "lightrag": LightRAGEfficiencyAdapter,
        "hyperrag_main": HyperRAGMainEfficiencyAdapter,
        "hyperrag_v81": HyperRAGV81EfficiencyAdapter,
    }
    if method_name not in adapters:
        raise ValueError(f"Unknown method: {method_name}")
    return adapters[method_name](contexts)


@dataclass
class MethodResult:
    method: str
    label: str
    retrieval_time_seconds: float | None = None
    generation_time_seconds: float | None = None
    generation_prompt_tokens: int | None = None
    answer_tokens: int | None = None
    generation_total_tokens: int | None = None
    context_tokens: int | None = None
    answer: str | None = None
    context_preview: str | None = None
    retrieval_meta: dict[str, Any] | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def run_method(method_name: str, question: str, contexts: list[str]) -> MethodResult:
    label = cfg.METHOD_LABELS.get(method_name, method_name)
    result = MethodResult(method=method_name, label=label)
    adapter = None
    try:
        print(f"\n[{label}] initializing index/cache...")
        adapter = create_adapter(method_name, contexts)

        print(f"[{label}] measuring retrieval...")
        t0 = time.perf_counter()
        context, meta = adapter.retrieve_context(question)
        result.retrieval_time_seconds = time.perf_counter() - t0
        result.retrieval_meta = meta
        result.context_tokens = count_tokens(context)
        result.context_preview = context[:1000]

        print(f"[{label}] measuring generation...")
        t0 = time.perf_counter()
        answer, usage = generate_answer(question, context)
        result.generation_time_seconds = time.perf_counter() - t0
        result.answer = answer
        result.generation_prompt_tokens = usage["generation_prompt_tokens"]
        result.answer_tokens = usage["answer_tokens"]
        result.generation_total_tokens = usage["generation_total_tokens"]
    except Exception as exc:
        result.error = str(exc)
    finally:
        if adapter is not None:
            adapter.cleanup()
    return result


def write_outputs(question: str, results: list[MethodResult]) -> tuple[Path, Path]:
    output_root = (
        Path(cfg.OUTPUT_DIR)
        / safe_cache_name(cfg.LLM_MODEL)
        / cfg.DATA_NAME
    )
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = f"{STAGE_CACHE_NAME}_q{cfg.QUESTION_INDEX + 1}"

    detail_file = output_root / f"efficiency_detail_{suffix}.json"
    csv_file = output_root / f"efficiency_summary_{suffix}.csv"

    detail = {
        "llm_model": cfg.LLM_MODEL,
        "embedding_model": cfg.EMB_MODEL,
        "dataset": cfg.DATA_NAME,
        "question_stage": cfg.QUESTION_STAGE,
        "question_index": cfg.QUESTION_INDEX,
        "question": question,
        "timestamp": datetime.now().isoformat(),
        "working_cache_dir": str(Path(cfg.WORKING_CACHE_DIR)),
        "cache_status": CACHE_STATUS_LOG,
        "results": [item.to_dict() for item in results],
    }
    with open(detail_file, "w", encoding="utf-8") as f:
        json.dump(detail, f, indent=2, ensure_ascii=False)

    fieldnames = [
        "method",
        "label",
        "retrieval_time_seconds",
        "generation_time_seconds",
        "context_tokens",
        "generation_prompt_tokens",
        "answer_tokens",
        "generation_total_tokens",
        "error",
    ]
    with open(csv_file, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for item in results:
            row = item.to_dict()
            writer.writerow({key: row.get(key) for key in fieldnames})

    return detail_file, csv_file


def print_summary(results: list[MethodResult]):
    print("\n" + "=" * 110)
    print("Efficiency summary")
    print("=" * 110)
    print(
        f"{'Method':<18}{'Retrieval(s)':>14}{'Generation(s)':>16}"
        f"{'ContextTok':>12}{'PromptTok':>12}{'AnswerTok':>12}{'TotalTok':>12}"
    )
    for item in results:
        if item.error:
            print(f"{item.label:<18} ERROR: {item.error}")
            continue
        print(
            f"{item.label:<18}"
            f"{item.retrieval_time_seconds:>14.4f}"
            f"{item.generation_time_seconds:>16.4f}"
            f"{item.context_tokens:>12}"
            f"{item.generation_prompt_tokens:>12}"
            f"{item.answer_tokens:>12}"
            f"{item.generation_total_tokens:>12}"
        )


def main():
    global CACHE_STATUS_LOG
    CACHE_STATUS_LOG = inspect_efficiency_caches()
    questions, contexts = load_questions_and_contexts()
    question = questions[cfg.QUESTION_INDEX]

    print("=" * 80)
    print("Retrieval time and generation token experiment")
    print(f"Model:    {cfg.LLM_MODEL}")
    print(f"Dataset:  {cfg.DATA_NAME}")
    print(f"Stage:    {cfg.QUESTION_STAGE}")
    print(f"Question: #{cfg.QUESTION_INDEX + 1} {question[:100]}")
    print(f"Cache:    {cfg.WORKING_CACHE_DIR}")
    print("=" * 80)
    for event in CACHE_STATUS_LOG:
        method = event.get("method", "cache")
        status = event.get("status", "unknown")
        file_count = event.get("file_count")
        if file_count is None:
            print(f"[cache] {method}: {status}")
        else:
            print(f"[cache] {method}: {status}, files={file_count}")

    results = [run_method(method, question, contexts) for method in cfg.RAG_METHODS]
    print_summary(results)
    detail_file, csv_file = write_outputs(question, results)
    print(f"\nDetail JSON: {detail_file}")
    print(f"Summary CSV: {csv_file}")


if __name__ == "__main__":
    main()
