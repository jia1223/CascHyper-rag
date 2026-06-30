# -*- coding: utf-8 -*-
"""
统一实验运行脚本
对比不同 RAG 方法 + 不同 LLM 的回答质量
使用 5 维度评分体系（Comprehensiveness, Diversity, Empowerment, Logical, Readability）
"""
import re
import sys
import json
import time
import hashlib
import asyncio
import importlib.util
import numpy as np
from pathlib import Path
from datetime import datetime
from openai import OpenAI

from ablation_config import (
    LLM_MODEL, LLM_BASE_URL, LLM_API_KEY,
    EVAL_MODEL, EVAL_BASE_URL, EVAL_API_KEY,
    EMB_MODEL, EMB_BASE_URL, EMB_API_KEY, EMB_DIM,
    DATA_NAME, QUESTION_STAGE, CACHES_DIR as CONFIG_CACHES_DIR,
    RAG_METHODS,
    HYPERRAG_MAIN_MODE,
    ABLATION_VARIANTS,
    SHARED_INDEX_VARIANTS,
    SHARED_INDEX_CACHE_PATH,
    RAG_V81_PATH as CONFIG_RAG_V81_PATH,
    HYPERRAG_MAIN_DIR as CONFIG_HYPERRAG_MAIN_DIR,
    LIGHTRAG_DIR as CONFIG_LIGHTRAG_DIR,
    SIMPLE_RAG_PATH as CONFIG_SIMPLE_RAG_PATH,
    OUTPUT_DIR as CONFIG_OUTPUT_DIR,
)

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CACHES_DIR = CONFIG_CACHES_DIR
RAG_V81_PATH = CONFIG_RAG_V81_PATH
HYPERRAG_MAIN_DIR = CONFIG_HYPERRAG_MAIN_DIR
LIGHTRAG_DIR = CONFIG_LIGHTRAG_DIR
SIMPLE_RAG_PATH = CONFIG_SIMPLE_RAG_PATH
OUTPUT_DIR = CONFIG_OUTPUT_DIR
STAGE_CACHE_NAME = f"{QUESTION_STAGE}_stage"


def safe_cache_name(value):
    """把模型名等配置值转换成稳定的缓存目录名。"""
    return re.sub(r'[^\w\-.]', '_', str(value))

# ============================================================================
# 重试与超时配置
# ============================================================================

OPENAI_MAX_RETRIES = 5          # OpenAI SDK 单次请求内部重试次数
OPENAI_TIMEOUT_SECONDS = 120
ADAPTER_INIT_MAX_RETRIES = 5    # RAG 初始化/索引整体重试次数
INDEX_MAX_RETRIES = 5           # 单次索引操作重试次数
QUERY_MAX_RETRIES = 5           # 每个问题查询重试次数
SCORING_MAX_RETRIES = 5         # 每个答案评分重试次数
RETRY_BASE_DELAY_SECONDS = 5
RETRY_MAX_DELAY_SECONDS = 30
ASYNC_CLEANUP_TIMEOUT_SECONDS = 10


def make_openai_client(api_key, base_url):
    """创建带有限重试和超时的 OpenAI 兼容客户端。"""
    return OpenAI(
        api_key=api_key,
        base_url=base_url,
        max_retries=OPENAI_MAX_RETRIES,
        timeout=OPENAI_TIMEOUT_SECONDS,
    )


def retry_delay(attempt):
    return min(RETRY_MAX_DELAY_SECONDS, RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1)))


def print_retry(label, attempt, max_attempts, error):
    delay = retry_delay(attempt)
    print(f"    ✗ {label} 失败 (第{attempt}/{max_attempts}次): {error}")
    print(f"    {delay:.1f}s 后重试...")
    time.sleep(delay)


def run_sync_with_retries(label, func, max_attempts):
    """同步操作有限重试，最终仍失败则抛出清晰错误。"""
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt == max_attempts:
                break
            print_retry(label, attempt, max_attempts, e)
    raise RuntimeError(f"{label} failed after {max_attempts} attempts: {last_error}") from last_error


def run_async_with_retries(loop, label, coro_factory, max_attempts=INDEX_MAX_RETRIES):
    """对异步索引/查询操作做有限重试；每次重试都重新创建 coroutine。"""
    return run_sync_with_retries(
        label,
        lambda: loop.run_until_complete(coro_factory()),
        max_attempts,
    )


async def _cancel_pending_tasks(tasks):
    for task in tasks:
        task.cancel()
    await asyncio.gather(*tasks, return_exceptions=True)


def close_event_loop_cleanly(loop, owner=None, finalize_methods=()):
    if loop is None or loop.is_closed():
        return

    asyncio.set_event_loop(loop)

    if owner is not None:
        for method_name in finalize_methods:
            method = getattr(owner, method_name, None)
            if not callable(method):
                continue
            try:
                result = method()
                if asyncio.iscoroutine(result):
                    loop.run_until_complete(result)
            except Exception as e:
                print(f"    [Warning] cleanup {method_name} failed: {e}")

    pending = [task for task in asyncio.all_tasks(loop) if not task.done()]
    if pending:
        try:
            loop.run_until_complete(
                asyncio.wait_for(
                    _cancel_pending_tasks(pending),
                    timeout=ASYNC_CLEANUP_TIMEOUT_SECONDS,
                )
            )
        except Exception as e:
            print(f"    [Warning] pending task cleanup failed: {e}")

    try:
        loop.run_until_complete(loop.shutdown_asyncgens())
    except Exception as e:
        print(f"    [Warning] async generator cleanup failed: {e}")

    try:
        loop.run_until_complete(loop.shutdown_default_executor())
    except Exception as e:
        print(f"    [Warning] async executor cleanup failed: {e}")

    asyncio.set_event_loop(None)
    loop.close()


# ============================================================================
# LLM 调用工具
# ============================================================================

def llm_call(prompt, system_prompt=None, model=None, base_url=None, api_key=None):
    """通用 LLM 调用"""
    client = make_openai_client(api_key or LLM_API_KEY, base_url or LLM_BASE_URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(
        model=model or LLM_MODEL, messages=messages
    )
    return response.choices[0].message.content


def eval_llm_call(prompt, system_prompt=None):
    """评判模型调用（固定 qwen-max）"""
    return llm_call(prompt, system_prompt, model=EVAL_MODEL,
                    base_url=EVAL_BASE_URL, api_key=EVAL_API_KEY)


# ============================================================================
# 数据加载
# ============================================================================

def load_questions_and_refs(caches_dir, data_name, stage):
    """从缓存目录加载问题和参考文档"""
    caches_root = Path(caches_dir)
    question_file = caches_root / data_name / "questions" / f"{stage}_stage.json"
    ref_file = caches_root / data_name / "questions" / f"{stage}_stage_ref.json"

    if not question_file.exists():
        print(f"[Error] 问题文件不存在: {question_file}")
        sys.exit(1)
    if not ref_file.exists():
        print(f"[Error] 参考文档文件不存在: {ref_file}")
        sys.exit(1)

    with open(question_file, "r", encoding="utf-8") as f:
        questions = json.load(f)
    with open(ref_file, "r", encoding="utf-8") as f:
        refs = json.load(f)

    print(f"  已加载 {len(questions)} 个问题, {len(refs)} 个参考文档")
    return questions, refs


# ============================================================================
# RAG 适配器
# ============================================================================

async def global_llm_func(prompt, system_prompt=None, **kwargs):
    history_messages = kwargs.pop("history_messages", []) or []
    hashing_kv = kwargs.pop("hashing_kv", None)
    model = kwargs.pop("model", LLM_MODEL)
    base_url = kwargs.pop("base_url", LLM_BASE_URL)
    api_key = kwargs.pop("api_key", LLM_API_KEY)

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    request_kwargs = {}
    for key in ("temperature", "top_p", "max_tokens", "presence_penalty", "frequency_penalty", "stop"):
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
    result = response.choices[0].message.content

    if hashing_kv is not None:
        await hashing_kv.upsert({cache_key: {"return": result, "model": model}})

    return result

async def global_emb_func_list(texts):
    client = make_openai_client(EMB_API_KEY, EMB_BASE_URL)
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    return [item.embedding for item in resp.data]

async def global_emb_func_ndarray(texts):
    client = make_openai_client(EMB_API_KEY, EMB_BASE_URL)
    resp = client.embeddings.create(model=EMB_MODEL, input=texts)
    import numpy as np
    return np.array([item.embedding for item in resp.data])

class BaseAdapter:
    """RAG 适配器基类"""
    name = "base"

    def __init__(self):
        pass

    def query(self, question, context=None):
        raise NotImplementedError

    def cleanup(self):
        pass


class LLMOnlyAdapter(BaseAdapter):
    """纯 LLM，不做任何检索"""
    name = "llm_only"

    def query(self, question, context=None):
        prompt = f"Please answer the following question:\n\n{question}"
        return llm_call(prompt)


class SimpleRAGAdapter(BaseAdapter):
    """普通 RAG 适配器"""
    name = "simple_rag"

    def __init__(self, contexts=None):
        super().__init__()
        self.contexts = contexts or []
        # 使用 simple_rag.py 的逻辑：对 contexts 做 embedding 然后检索
        self._init_rag()

    def _init_rag(self):
        """初始化简单 RAG（基于向量检索，严格通过 token 长度切分保证公平）"""
        import tiktoken
        self.emb_client = make_openai_client(EMB_API_KEY, EMB_BASE_URL)
        
        self.chunks = []
        self.embeddings = []
        chunk_token_size = 1200
        overlap_token_size = 100
        
        try:
            encoder = tiktoken.encoding_for_model("gpt-3.5-turbo")
        except KeyError:
            encoder = tiktoken.get_encoding("cl100k_base")

        for ctx in self.contexts:
            tokens = encoder.encode(ctx)
            if len(tokens) <= chunk_token_size:
                self.chunks.append(ctx)
                continue
                
            start = 0
            while start < len(tokens):
                end = min(start + chunk_token_size, len(tokens))
                chunk_tokens = tokens[start:end]
                self.chunks.append(encoder.decode(chunk_tokens))
                start += (chunk_token_size - overlap_token_size)

        if self.chunks:
            print(f"    SimpleRAG: 编码 {len(self.chunks)} 个文本块...")
            # 批量编码（每批 20 个）
            batch_size = 20
            all_embs = []
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i:i + batch_size]
                try:
                    resp = run_sync_with_retries(
                        f"SimpleRAG embedding batch {i // batch_size + 1}",
                        lambda: self.emb_client.embeddings.create(model=EMB_MODEL, input=batch),
                        INDEX_MAX_RETRIES,
                    )
                    for item in resp.data:
                        all_embs.append(item.embedding)
                except Exception as e:
                    raise RuntimeError(f"SimpleRAG embedding failed while indexing batch {i // batch_size + 1}: {e}") from e
            self.embeddings = np.array(all_embs)
            # 归一化
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.embeddings = self.embeddings / norms

    def _get_embedding(self, text):
        try:
            resp = run_sync_with_retries(
                "SimpleRAG query embedding",
                lambda: self.emb_client.embeddings.create(model=EMB_MODEL, input=[text]),
                QUERY_MAX_RETRIES,
            )
            emb = np.array(resp.data[0].embedding)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception as e:
            raise RuntimeError(f"SimpleRAG query embedding failed: {e}") from e

    def query(self, question, context=None):
        if len(self.chunks) == 0:
            return llm_call(f"Please answer the following question:\n\n{question}")

        # 检索 top-5 相关文本块
        q_emb = self._get_embedding(question)
        scores = self.embeddings @ q_emb
        top_k = min(5, len(self.chunks))
        top_indices = np.argsort(scores)[-top_k:][::-1]
        retrieved = "\n\n".join([self.chunks[i] for i in top_indices])

        prompt = f"""Based on the following retrieved context, please answer the question.

Context:
{retrieved}

Question:
{question}

Answer:"""
        return llm_call(prompt)


class HyperRAGAblationAdapter(BaseAdapter):
    """HyperRAG v8.1 ablation adapter."""
    name = "hyperrag_ablation"

    def __init__(self, contexts=None, variant_name="full_hyperrag_v81"):
        super().__init__()
        self.contexts = contexts or []
        self.variant_name = variant_name
        self.variant_config = dict(ABLATION_VARIANTS[variant_name])
        self.variant_config["name"] = variant_name
        self.rag_engine = None
        self.init_error = None
        self.loop = asyncio.new_event_loop()
        self._init_rag()

    def _get_index_cache_path(self, raw_text, get_cache_path):
        if self.variant_name in SHARED_INDEX_VARIANTS:
            shared_path = Path(SHARED_INDEX_CACHE_PATH)
            if shared_path.exists():
                return str(shared_path), "shared explicit full-index cache"

            # Fallback for newly generated shared caches: use the historical
            # full-index prefix instead of the variant name so query-time
            # ablations do not rebuild identical document/entity indexes.
            return (
                get_cache_path(raw_text, prefix="hyperrag_v81", model=LLM_MODEL),
                "shared full-index cache",
            )

        cache_prefix = f"{self.variant_name}_{STAGE_CACHE_NAME}"
        return (
            get_cache_path(raw_text, prefix=cache_prefix, model=LLM_MODEL),
            "variant-specific cache",
        )

    def _init_rag(self):
        """动态加载 rag(8.1).py 并初始化引擎（支持 pkl 缓存）"""
        try:
            spec = importlib.util.spec_from_file_location("rag_v81", RAG_V81_PATH)
            rag_module = importlib.util.module_from_spec(spec)
            sys.modules["rag_v81"] = rag_module
            spec.loader.exec_module(rag_module)

            # 获取 HyperRAG 类和缓存工具函数
            rag_class = getattr(rag_module, "HyperRAG_Attention_v81", None)
            get_cache_path = getattr(rag_module, "get_cache_path", None)
            load_engine_state = getattr(rag_module, "load_engine_state", None)
            save_engine_state = getattr(rag_module, "save_engine_state", None)

            if rag_class is None:
                # 尝试其它可能的类名
                for attr_name in dir(rag_module):
                    attr = getattr(rag_module, attr_name)
                    if isinstance(attr, type) and attr_name.lower().startswith("hyperrag"):
                        rag_class = attr
                        break

            if rag_class:
                self.rag_engine = rag_class(ablation_config=self.variant_config)
                # 构建索引
                if self.contexts:
                    raw_text = "\n\n".join(self.contexts)
                    asyncio.set_event_loop(self.loop)

                    # 尝试从缓存加载引擎状态
                    cache_loaded = False
                    cache_path = None
                    cache_desc = "cache"
                    if get_cache_path and load_engine_state:
                        cache_path, cache_desc = self._get_index_cache_path(raw_text, get_cache_path)
                        cache_loaded = load_engine_state(self.rag_engine, cache_path)

                    if cache_loaded:
                        print(f"    {self.variant_name}: 从 {cache_desc} 加载，跳过索引构建")
                    else:
                        print(f"    {self.variant_name}: 索引 {len(self.contexts)} 个文档...")
                        run_async_with_retries(
                            self.loop,
                            f"{self.variant_name} index build",
                            lambda: self.rag_engine.build_index(
                                raw_text,
                                max_token_size=1200,
                                overlap_token_size=100
                            ),
                        )
                        # 保存缓存供下次使用
                        if save_engine_state and cache_path:
                            save_engine_state(self.rag_engine, cache_path)

                print(f"    {self.variant_name}: 初始化完成 ({self.variant_config.get('label', self.variant_name)})")
            else:
                raise RuntimeError("HyperRAG v8.1 class not found")
        except Exception as e:
            self.init_error = str(e)
            self.rag_engine = None
            raise RuntimeError(f"{self.variant_name} initialization failed: {e}") from e

    def query(self, question, context=None):
        if self.rag_engine is None:
            raise RuntimeError(f"HyperRAG v8.1 initialization failed: {self.init_error or 'unknown error'}")
        try:
            asyncio.set_event_loop(self.loop)
            # 按照 v8.1 标准流程: search -> verify -> generate
            results = self.loop.run_until_complete(
                self.rag_engine.search(
                    question,
                    top_k_chunks=5,
                    top_k_sents=10,
                    enable_multi_hop=self.variant_config.get("enable_multi_hop", True),
                )
            )
            results = self.loop.run_until_complete(self.rag_engine.verify_results(question, results))
            result = self.loop.run_until_complete(self.rag_engine.generate_answer(question, results))
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            raise RuntimeError(f"{self.variant_name} query failed: {e}") from e

    def cleanup(self):
        close_event_loop_cleanly(self.loop, self.rag_engine)


class HyperRAGMainAdapter(BaseAdapter):
    """Hyper-RAG-main 适配器"""
    name = "hyperrag_main"

    def __init__(self, contexts=None):
        super().__init__()
        self.contexts = contexts or []
        self.rag_instance = None
        self.loop = asyncio.new_event_loop()
        self._init_rag()

    def _init_rag(self):
        """加载 Hyper-RAG-main 的 HyperRAG 类"""
        try:
            # rag(8.1).py also imports a package named "hyperrag" from ./Hyper-RAG.
            # Clear it before loading Hyper-RAG-main so Python does not reuse the
            # wrong package version from sys.modules.
            for module_name in list(sys.modules):
                if module_name == "hyperrag" or module_name.startswith("hyperrag."):
                    del sys.modules[module_name]
            main_dir = str(HYPERRAG_MAIN_DIR)
            if main_dir in sys.path:
                sys.path.remove(main_dir)
            sys.path.insert(0, main_dir)
            from hyperrag import HyperRAG

            # 按模型、数据集、问题阶段区分缓存目录，不同阶段的问题缓存互不污染
            safe_model = safe_cache_name(LLM_MODEL)
            working_dir = HYPERRAG_MAIN_DIR / "caches" / safe_model / DATA_NAME / STAGE_CACHE_NAME
            working_dir.mkdir(parents=True, exist_ok=True)

            from hyperrag.utils import EmbeddingFunc
            wrapped_emb_func = EmbeddingFunc(embedding_dim=EMB_DIM, max_token_size=8192, func=global_emb_func_list)

            self.rag_instance = HyperRAG(
                working_dir=str(working_dir),
                llm_model_func=global_llm_func,
                embedding_func=wrapped_emb_func,
                chunk_token_size=1200,
                chunk_overlap_token_size=100,
            )

            if self.contexts:
                print(f"    Hyper-RAG-main: 索引 {len(self.contexts)} 个文档...")
                asyncio.set_event_loop(self.loop)
                run_async_with_retries(
                    self.loop,
                    "Hyper-RAG-main index build",
                    lambda: self.rag_instance.ainsert(self.contexts),
                )
            print("    Hyper-RAG-main: 初始化完成")
        except Exception as e:
            self.rag_instance = None
            raise RuntimeError(f"Hyper-RAG-main initialization failed: {e}") from e

    def query(self, question, context=None):
        if self.rag_instance is None:
            raise RuntimeError("Hyper-RAG-main is not initialized")
        try:
            asyncio.set_event_loop(self.loop)
            from hyperrag.base import QueryParam
            param = QueryParam(mode=HYPERRAG_MAIN_MODE, top_k=10, max_token_for_text_unit=1200) # Added equivalent param
            # Note: HyperRAG QueryParam natively uses `max_token_for_text_unit` instead of `chunk_top_k` for chunk size, top_k applies across all retrieved structures.
            result = self.loop.run_until_complete(self.rag_instance.aquery(question, param))
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            raise RuntimeError(f"Hyper-RAG-main query failed: {e}") from e

    def cleanup(self):
        close_event_loop_cleanly(self.loop, self.rag_instance)


class LightRAGAdapter(BaseAdapter):
    """LightRAG 适配器"""
    name = "lightrag"

    def __init__(self, contexts=None):
        super().__init__()
        self.contexts = contexts or []
        self.rag_instance = None
        self.loop = asyncio.new_event_loop()
        self._init_rag()

    def _init_rag(self):
        """加载 LightRAG"""
        try:
            sys.path.insert(0, str(LIGHTRAG_DIR))
            from lightrag import LightRAG as _LightRAG, QueryParam
            from lightrag.utils import EmbeddingFunc

            # 按模型、数据集、问题阶段区分缓存目录，不同阶段的问题缓存互不污染
            safe_model = safe_cache_name(LLM_MODEL)
            working_dir = LIGHTRAG_DIR / "caches" / safe_model / DATA_NAME / STAGE_CACHE_NAME
            working_dir.mkdir(parents=True, exist_ok=True)

            self.rag_instance = _LightRAG(
                working_dir=str(working_dir),
                llm_model_func=global_llm_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=EMB_DIM,
                    max_token_size=8192,
                    func=global_emb_func_ndarray
                ),
                chunk_token_size=1200,
                chunk_overlap_token_size=100,
            )

            if self.contexts:
                print(f"    LightRAG: 索引 {len(self.contexts)} 个文档...")
                asyncio.set_event_loop(self.loop)
                self.loop.run_until_complete(self.rag_instance.initialize_storages())
                run_async_with_retries(
                    self.loop,
                    "LightRAG index build",
                    lambda: self.rag_instance.ainsert(self.contexts),
                )
            else:
                asyncio.set_event_loop(self.loop)
                self.loop.run_until_complete(self.rag_instance.initialize_storages())
            print("    LightRAG: 初始化完成")
        except Exception as e:
            self.rag_instance = None
            raise RuntimeError(f"LightRAG initialization failed: {e}") from e

    def query(self, question, context=None):
        if self.rag_instance is None:
            raise RuntimeError("LightRAG is not initialized")
        try:
            asyncio.set_event_loop(self.loop)
            from lightrag import QueryParam
            param = QueryParam(mode="mix", chunk_top_k=5, top_k=10)
            result = self.loop.run_until_complete(self.rag_instance.aquery(question, param))
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            raise RuntimeError(f"LightRAG query failed: {e}") from e

    def cleanup(self):
        close_event_loop_cleanly(
            self.loop,
            self.rag_instance,
            finalize_methods=("finalize_storages",),
        )


# ============================================================================
# 评分系统（复用 step_4_evaluate_scoring.py 的逻辑）
# ============================================================================

CORE_METRICS = ["Comprehensiveness", "Diversity", "Empowerment", "Logical", "Readability"]
RELEVANCE_METRIC = "Relevance"
ALL_METRICS = CORE_METRICS + [RELEVANCE_METRIC]

SCORING_SYSTEM_PROMPT = """---Role---
You are an expert tasked with evaluating answers to the questions by using the relevant documents based on six criteria:**Comprehensiveness**, **Diversity**, **Empowerment**, **Logical**, **Readability**, and **Relevance**."""

SCORING_PROMPT_TEMPLATE = """You will evaluate the answers to the questions by using the relevant documents based on six criteria:**Comprehensiveness**, **Diversity**, **Empowerment**, **Logical**, **Readability**, and **Relevance**.

- **Comprehensiveness** -
Measure whether the answer comprehensively covers all key aspects of the question and whether there are omissions.
Level   | score range | description
Level 1 | 0-20   | The answer is extremely one-sided, leaving out key parts or important aspects of the question.
Level 2 | 20-40  | The answer has some content, but it misses many important aspects of the question and is not comprehensive enough.
Level 3 | 40-60  | The answer is more comprehensive, covering the main aspects of the question, but there are still some omissions.
Level 4 | 60-80  | The answer is comprehensive, covering most aspects of the question, with few omissions.
Level 5 | 80-100 | The answer is extremely comprehensive, covering all aspects of the question with no omissions.

- **Diversity** -
Measure the richness of the answer content, including background knowledge, extended information, case studies, etc.
Level   | score range | description
Level 1 | 0-20   | The answer is extremely sparse, providing only direct answers without additional information.
Level 2 | 20-40  | The answer provides a direct answer but contains only a small amount of relevant knowledge expansion.
Level 3 | 40-60  | In addition to the direct answers, the answer also provides some relevant background knowledge.
Level 4 | 60-80  | The answer is rich, providing more relevant background knowledge and supplementary information.
Level 5 | 80-100 | The answer provides a lot of relevant knowledge, expanded content and in-depth analysis.

- **Empowerment** -
Measure the credibility of the answer and whether it convinces the reader that it is correct.
Level   | score range | description
Level 1 | 0-20   | The answer lacks credibility, contains obvious errors or false information.
Level 2 | 20-40  | The answer has some credibility, but some information is not accurate.
Level 3 | 40-60  | The answer is credible and provides some supporting information.
Level 4 | 60-80  | The answer is highly credible, providing sufficient supporting information.
Level 5 | 80-100 | The answer is highly credible with authoritative supporting information.

- **Logical** -
Measure whether the answers are coherent, clear, and easy to understand.
Level   | score range | description
Level 1 | 0-20   | The answer is illogical, incoherent, and difficult to understand.
Level 2 | 20-40  | The answer has some logic, but is incoherent in parts.
Level 3 | 40-60  | The answer is logically clear and basically coherent.
Level 4 | 60-80  | The answer is logical, coherent, and easy to understand.
Level 5 | 80-100 | The answer is extremely logical, fluent and well-organized.

- **Readability** -
Measure whether the answer is well organized, clear in format, and easy to read.
Level   | score range | description
Level 1 | 0-20   | The format is confused and difficult to read.
Level 2 | 20-40  | There are some problems in the format.
Level 3 | 40-60  | The format is basically clear.
Level 4 | 60-80  | The format is clear and well organized.
Level 5 | 80-100 | The format is very clear with excellent reading experience.

- **Relevance** -
Measure whether the reasoning and answer are highly relevant and helpful to the question.
Level   | score range | description
Level 1 | 0-20   | The answer is entirely irrelevant, barely related to the question, or largely unhelpful.
Level 2 | 20-40  | The answer has limited relevance; much of the response is off-topic or unhelpful.
Level 3 | 40-60  | The answer is generally relevant, but includes distractions or less helpful parts.
Level 4 | 60-80  | The answer is mostly on point, with only minor digressions and overall useful content.
Level 5 | 80-100 | The answer is fully focused on the question, highly relevant, and highly helpful.

For each indicator, give a Level and a Score within the level's score range.

Here are the relevant documents:
    {reference}

Here are the questions:
    {query}

Here are the answers:
    {answer}

Output your evaluation in the following JSON format:
{{
    "Comprehensiveness": {{
        "Explanation": "Provide explanation here",
        "Level": "A single number 1-5",
        "Score": "A single number 0-100"
    }},
    "Diversity": {{
        "Explanation": "Provide explanation here",
        "Level": "A single number 1-5",
        "Score": "A single number 0-100"
    }},
    "Empowerment": {{
        "Explanation": "Provide explanation here",
        "Level": "A single number 1-5",
        "Score": "A single number 0-100"
    }},
    "Logical": {{
        "Explanation": "Provide explanation here",
        "Level": "A single number 1-5",
        "Score": "A single number 0-100"
    }},
    "Readability": {{
        "Explanation": "Provide explanation here",
        "Level": "A single number 1-5",
        "Score": "A single number 0-100"
    }},
    "Relevance": {{
        "Explanation": "Provide explanation here",
        "Level": "A single number 1-5",
        "Score": "A single number 0-100"
    }}
}}"""

RELEVANCE_SYSTEM_PROMPT = """---Role---
You are an expert tasked with evaluating whether an answer is relevant and helpful to the question."""

RELEVANCE_PROMPT_TEMPLATE = """Evaluate only the Relevance of the answer to the question.

- **Relevance** -
Measure whether the reasoning and answer are highly relevant and helpful to the question.
Level   | score range | description
Level 1 | 0-20   | The answer is entirely irrelevant, barely related to the question, or largely unhelpful.
Level 2 | 20-40  | The answer has limited relevance; much of the response is off-topic or unhelpful.
Level 3 | 40-60  | The answer is generally relevant, but includes distractions or less helpful parts.
Level 4 | 60-80  | The answer is mostly on point, with only minor digressions and overall useful content.
Level 5 | 80-100 | The answer is fully focused on the question, highly relevant, and highly helpful.

Here are the relevant documents:
    {reference}

Here is the question:
    {query}

Here is the answer:
    {answer}

Output your evaluation in the following JSON format:
{{
    "Relevance": {{
        "Explanation": "Provide explanation here",
        "Level": "A single number 1-5",
        "Score": "A single number 0-100"
    }}
}}"""


def score_single_answer(query, answer, reference):
    """对单个回答进行 6 维度评分"""
    prompt = SCORING_PROMPT_TEMPLATE.format(
        reference=reference, query=query, answer=answer
    )
    try:
        response = run_sync_with_retries(
            "scoring LLM call",
            lambda: eval_llm_call(prompt, SCORING_SYSTEM_PROMPT),
            SCORING_MAX_RETRIES,
        )
        return response
    except Exception as e:
        print(f"    [Error] 评分失败: {e}")
        return "{}"


def score_relevance_only(query, answer, reference):
    prompt = RELEVANCE_PROMPT_TEMPLATE.format(
        reference=reference, query=query, answer=answer
    )
    try:
        return run_sync_with_retries(
            "relevance scoring LLM call",
            lambda: eval_llm_call(prompt, RELEVANCE_SYSTEM_PROMPT),
            SCORING_MAX_RETRIES,
        )
    except Exception as e:
        print(f"    [Error] 相关性评分失败: {e}")
        return "{}"


def _extract_json_payload(response_text):
    if not isinstance(response_text, str):
        return None

    candidates = re.findall(r"```(?:json)?\s*(.*?)```", response_text, flags=re.DOTALL)
    candidates.append(response_text)
    for candidate in candidates:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start == -1 or end == -1 or start >= end:
            continue
        try:
            payload = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def _parse_metric_score(response_text, metric):
    payload = _extract_json_payload(response_text)
    if payload and isinstance(payload.get(metric), dict):
        value = payload[metric].get("Score")
        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    pattern = rf'"{re.escape(metric)}"\s*:\s*\{{.*?"Score"\s*:\s*"?(\d+(?:\.\d+)?)"?'
    match = re.search(pattern, response_text or "", flags=re.DOTALL)
    if match:
        return float(match.group(1))
    return None


def parse_scores(response_text, metrics=ALL_METRICS):
    """解析评分。"""
    scores = {}
    for metric in metrics:
        value = _parse_metric_score(response_text, metric)
        if value is None:
            continue
        scores[metric] = value
    return scores


def merge_scoring_responses(response_text, relevance_response_text=None):
    scores = parse_scores(response_text)
    if RELEVANCE_METRIC not in scores and relevance_response_text is not None:
        scores.update(parse_scores(relevance_response_text, metrics=[RELEVANCE_METRIC]))
    return scores


def scoring_cache_has_relevance(scoring_data, expected_count):
    raw_responses = scoring_data.get("raw_responses", [])
    if len(raw_responses) != expected_count:
        return False

    if all(_parse_metric_score(resp, RELEVANCE_METRIC) is not None for resp in raw_responses):
        return True

    relevance_raw_responses = scoring_data.get("relevance_raw_responses", [])
    return (
        len(relevance_raw_responses) == expected_count
        and all(
            _parse_metric_score(resp, RELEVANCE_METRIC) is not None
            for resp in relevance_raw_responses
        )
    )


def build_parsed_score_rows(scoring_data):
    raw_responses = scoring_data.get("raw_responses", [])
    relevance_raw_responses = scoring_data.get("relevance_raw_responses", [])
    rows = []
    for idx, response_text in enumerate(raw_responses):
        relevance_response = (
            relevance_raw_responses[idx]
            if idx < len(relevance_raw_responses)
            else None
        )
        rows.append(merge_scoring_responses(response_text, relevance_response))
    return rows


def compute_aggregate_scores(all_scores):
    """计算所有问题的平均分"""
    totals = {m: 0.0 for m in ALL_METRICS}
    valid_count = 0

    for scores in all_scores:
        if all(metric in scores for metric in ALL_METRICS):
            for m in ALL_METRICS:
                totals[m] += scores.get(m, 0)
            valid_count += 1

    if valid_count == 0:
        return {m: 0.0 for m in ALL_METRICS + ["Average"]}

    result = {m: totals[m] / valid_count for m in ALL_METRICS}
    result["Average"] = sum(result.values()) / len(ALL_METRICS)
    return result


# ============================================================================
# 主实验流程
# ============================================================================

def create_adapter(method_name, contexts):
    """Create the adapter for an ablation variant or an optional baseline."""
    if method_name in ABLATION_VARIANTS:
        return HyperRAGAblationAdapter(contexts, variant_name=method_name)

    """根据方法名创建对应的适配器"""
    adapters = {
        "llm_only": LLMOnlyAdapter,
        "simple_rag": lambda: SimpleRAGAdapter(contexts),
        "hyperrag_v81": lambda: HyperRAGAblationAdapter(contexts, variant_name="full_hyperrag_v81"),
        "hyperrag_main": lambda: HyperRAGMainAdapter(contexts),
        "lightrag": lambda: LightRAGAdapter(contexts),
    }
    factory = adapters.get(method_name)
    if factory is None:
        raise ValueError(f"Unknown RAG method: {method_name}")
    if callable(factory) and not isinstance(factory, type):
        return factory()
    return factory()


def create_adapter_with_retries(method_name, contexts):
    """初始化 RAG 适配器；索引阶段失败时有限重试，最终失败后交给方法级处理。"""
    last_error = None
    last_attempt = 0
    for attempt in range(1, ADAPTER_INIT_MAX_RETRIES + 1):
        last_attempt = attempt
        try:
            return create_adapter(method_name, contexts)
        except Exception as e:
            last_error = e
            # 索引操作内部已经做过 INDEX_MAX_RETRIES 次，避免再整套重建多轮。
            nested_index_retry_exhausted = "index build failed after" in str(e)
            if attempt == ADAPTER_INIT_MAX_RETRIES or nested_index_retry_exhausted:
                break
            print_retry(f"{method_name} adapter initialization", attempt, ADAPTER_INIT_MAX_RETRIES, e)
    raise RuntimeError(
        f"{method_name} adapter initialization failed after {last_attempt} attempts: {last_error}"
    ) from last_error


def run_experiment():
    """运行完整实验"""
    print("=" * 70)
    print(f"  统一实验评估框架")
    print(f"  LLM Model:      {LLM_MODEL}")
    print(f"  Eval Model:      {EVAL_MODEL}")
    print(f"  Dataset:         {DATA_NAME}")
    print(f"  Question Stage:  {QUESTION_STAGE}")
    print(f"  RAG Methods:     {', '.join(RAG_METHODS)}")
    print(f"  Time:            {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # 1. 加载问题和参考文档
    print("\n[Step 1] 加载问题和参考文档...")
    questions, refs = load_questions_and_refs(CACHES_DIR, DATA_NAME, QUESTION_STAGE)

    # 2. 加载 contexts（用于建立 RAG 索引）
    print("\n[Step 2] 加载文档 contexts...")
    contexts_file = CACHES_DIR / DATA_NAME / "contexts" / f"{DATA_NAME}_unique_contexts.json"
    contexts = []
    if contexts_file.exists():
        with open(contexts_file, "r", encoding="utf-8") as f:
            contexts = json.load(f)
        print(f"  已加载 {len(contexts)} 个 context 文档")
    else:
        print(f"  [Warning] Context 文件不存在: {contexts_file}")

    # 3. 准备输出目录
    output_root = OUTPUT_DIR / LLM_MODEL / DATA_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    # 4. 对每个 RAG 方法运行实验
    all_results = {}
    method_errors = {}

    for method_name in RAG_METHODS:
        print(f"\n{'='*60}")
        print(f"[Step 3] 运行 RAG 方法: {method_name}")
        print(f"{'='*60}")

        # 检查是否已有缓存的回答结果
        response_file = output_root / f"{method_name}_{QUESTION_STAGE}_stage_result.json"
        scoring_file = output_root / f"{method_name}_{QUESTION_STAGE}_stage_scoring.json"

        try:
            # 3a. 生成回答
            if response_file.exists():
                print(f"  已有缓存结果，加载: {response_file}")
                with open(response_file, "r", encoding="utf-8") as f:
                    answers_data = json.load(f)
                answers = [item["result"] for item in answers_data]
                if len(answers_data) != len(questions):
                    response_file.unlink()
                    if scoring_file.exists():
                        scoring_file.unlink()
                    raise RuntimeError(
                        f"{method_name} cached answer count mismatch "
                        f"({len(answers_data)}/{len(questions)}); deleted invalid caches, rerun will regenerate"
                    )

                # 检查是否有失败的题目需要重试；失败缓存不能直接复用
                failed_indices = [
                    i for i, a in enumerate(answers)
                    if isinstance(a, str) and a.startswith("[Error]")
                ]
                if failed_indices:
                    print(f"  发现 {len(failed_indices)} 个失败题目，开始重试并更新缓存: {[i+1 for i in failed_indices]}")
                    adapter = create_adapter_with_retries(method_name, contexts)
                    try:
                        for idx in failed_indices:
                            question = questions[idx]
                            success = False
                            last_error = None
                            for attempt in range(1, QUERY_MAX_RETRIES + 1):
                                print(f"  重试 [{idx+1}/{len(questions)}] (第{attempt}次) {question[:60]}...")
                                try:
                                    answer = adapter.query(question)
                                    if answer.startswith("[Error]"):
                                        raise RuntimeError(answer)
                                    answers[idx] = answer
                                    answers_data[idx] = {"query": question, "result": answer}
                                    print(f"    ✓ 重试成功")
                                    success = True
                                    break
                                except Exception as e:
                                    last_error = e
                                    if attempt < QUERY_MAX_RETRIES:
                                        print_retry(
                                            f"{method_name} cached question {idx+1}",
                                            attempt,
                                            QUERY_MAX_RETRIES,
                                            e,
                                        )
                            if not success:
                                raise RuntimeError(
                                    f"{method_name} failed on cached question {idx+1} after {QUERY_MAX_RETRIES} retries: {last_error}"
                                ) from last_error
                    finally:
                        adapter.cleanup()

                    still_failed = [
                        i for i, a in enumerate(answers)
                        if isinstance(a, str) and a.startswith("[Error]")
                    ]
                    if still_failed:
                        raise RuntimeError(
                            f"{method_name} still has failed cached answers: {[i+1 for i in still_failed]}"
                        )

                    with open(response_file, "w", encoding="utf-8") as f:
                        json.dump(answers_data, f, indent=2, ensure_ascii=False)
                    print(f"  重试结果已更新: {response_file}")
                    if scoring_file.exists():
                        scoring_file.unlink()
                        print(f"  已删除旧评分缓存，将重新评分")
                    print(f"  ✓ 所有失败题目已重试成功")
                else:
                    print(f"  所有题目均有有效回答，无需重试")
            else:
                print(f"  初始化 {method_name}...")
                adapter = create_adapter_with_retries(method_name, contexts)

                answers = []
                answers_data = []
                try:
                    for i, question in enumerate(questions):
                        print(f"  [{i+1}/{len(questions)}] {question[:60]}...")
                        for attempt in range(1, QUERY_MAX_RETRIES + 1):
                            if attempt > 1:
                                print(f"    重试 (第{attempt}次)...")
                            try:
                                answer = adapter.query(question)
                                if answer.startswith("[Error]"):
                                    raise RuntimeError(answer)
                                answers.append(answer)
                                answers_data.append({
                                    "query": question,
                                    "result": answer,
                                })
                                break
                            except Exception as e:
                                if attempt == QUERY_MAX_RETRIES:
                                    raise RuntimeError(
                                        f"{method_name} failed on question {i+1} after {QUERY_MAX_RETRIES} attempts: {e}"
                                    ) from e
                                print_retry(f"{method_name} question {i+1}", attempt, QUERY_MAX_RETRIES, e)
                finally:
                    adapter.cleanup()

                # 只保存完整成功的回答，避免下次运行读到半成品缓存
                if len(answers_data) != len(questions):
                    raise RuntimeError(
                        f"{method_name} produced {len(answers_data)}/{len(questions)} answers; incomplete result cache was not saved"
                    )
                with open(response_file, "w", encoding="utf-8") as f:
                    json.dump(answers_data, f, indent=2, ensure_ascii=False)
                print(f"  回答已保存: {response_file}")

            failed_answers = [
                i for i, a in enumerate(answers)
                if isinstance(a, str) and a.startswith("[Error]")
            ]
            if failed_answers:
                raise RuntimeError(
                    f"{method_name} has failed answers and will not be scored: {[i+1 for i in failed_answers]}"
                )

            # 3b. 评分
            if scoring_file.exists():
                print(f"  已有缓存评分，加载: {scoring_file}")
                with open(scoring_file, "r", encoding="utf-8") as f:
                    scoring_data = json.load(f)
                if len(scoring_data.get("raw_responses", [])) != len(questions):
                    scoring_file.unlink()
                    raise RuntimeError(
                        f"{method_name} cached scoring count mismatch "
                        f"({len(scoring_data.get('raw_responses', []))}/{len(questions)}); "
                        "deleted invalid scoring cache, rerun will regenerate"
                    )

                if not scoring_cache_has_relevance(scoring_data, len(questions)):
                    print(f"  缓存评分缺少 Relevance，开始只补相关性评分...")
                    relevance_raw_responses = []
                    for i, (q, a, r) in enumerate(zip(questions, answers, refs)):
                        print(f"  相关性补评 [{i+1}/{len(questions)}]...")
                        relevance_raw_responses.append(score_relevance_only(q, a, r))
                    scoring_data["relevance_raw_responses"] = relevance_raw_responses
                    scoring_data["metrics"] = ALL_METRICS
                    scoring_data["relevance_raw_scale"] = "0-100"
                    scoring_data["aggregate_relevance_scale"] = "0-100"
                    with open(scoring_file, "w", encoding="utf-8") as f:
                        json.dump(scoring_data, f, indent=2, ensure_ascii=False)
                    print(f"  已补充 Relevance 评分: {scoring_file}")
                else:
                    print(f"  缓存中已包含 Relevance，跳过补评")

                all_parsed_scores = build_parsed_score_rows(scoring_data)
            else:
                print(f"\n  开始评分 ({method_name})...")
                raw_responses = []
                all_parsed_scores = []
                for i, (q, a, r) in enumerate(zip(questions, answers, refs)):
                    print(f"  评分 [{i+1}/{len(questions)}]...")
                    resp = score_single_answer(q, a, r)
                    raw_responses.append(resp)
                    scores = merge_scoring_responses(resp)
                    all_parsed_scores.append(scores)

                scoring_data = {
                    "method": method_name,
                    "model": LLM_MODEL,
                    "eval_model": EVAL_MODEL,
                    "question_stage": QUESTION_STAGE,
                    "raw_responses": raw_responses,
                    "metrics": ALL_METRICS,
                    "relevance_raw_scale": "0-100",
                    "aggregate_relevance_scale": "0-100",
                }
                with open(scoring_file, "w", encoding="utf-8") as f:
                    json.dump(scoring_data, f, indent=2, ensure_ascii=False)
                print(f"  评分已保存: {scoring_file}")

            # 计算汇总分数
            aggregate = compute_aggregate_scores(all_parsed_scores)
            all_results[method_name] = aggregate

            print(f"\n  {method_name} 汇总分数:")
            for metric, score in aggregate.items():
                print(f"    {metric:20}: {score:.2f}")
        except Exception as e:
            method_errors[method_name] = str(e)
            if scoring_file.exists():
                scoring_file.unlink()
                print(f"  已删除 {method_name} 的旧评分缓存，避免失败方法继续复用旧分数")
            print(f"  [Method Failed] {method_name}: {e}")
            print(f"  继续运行下一个 RAG 方法...")
            continue

    # 5. 输出总汇表
    print(f"\n{'='*70}")
    print(f"  实验结果汇总  (LLM: {LLM_MODEL}, Dataset: {DATA_NAME}, Stage: {QUESTION_STAGE})")
    print(f"{'='*70}")

    metrics = ALL_METRICS + ["Average"]

    # 表头
    header = f"{'Method':<20}"
    for m in metrics:
        header += f"{m:>16}"
    print(header)
    print("-" * (20 + 16 * len(metrics)))

    # 数据行
    for method, scores in all_results.items():
        row = f"{method:<20}"
        for m in metrics:
            row += f"{scores.get(m, 0):>16.2f}"
        print(row)

    if method_errors:
        print(f"\n失败的 RAG 方法:")
        for method, error in method_errors.items():
            print(f"  - {method}: {error}")

    # 保存汇总到 JSON
    summary = {
        "llm_model": LLM_MODEL,
        "eval_model": EVAL_MODEL,
        "dataset": DATA_NAME,
        "question_stage": QUESTION_STAGE,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
        "errors": method_errors,
    }
    summary_file = output_root / f"summary_{STAGE_CACHE_NAME}.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_file}")

    return all_results


if __name__ == "__main__":
    run_experiment()
