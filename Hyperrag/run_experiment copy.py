# -*- coding: utf-8 -*-
"""
统一实验运行脚本
对比不同 RAG 方法 + 不同 LLM 的回答质量
使用 5 维度评分体系（Comprehensiveness, Diversity, Empowerment, Logical, Readability）
"""
import re
import sys
import json
import asyncio
import importlib.util
import numpy as np
from pathlib import Path
from datetime import datetime
from openai import OpenAI

from experiment_config import (
    LLM_MODEL, LLM_BASE_URL, LLM_API_KEY,
    EVAL_MODEL, EVAL_BASE_URL, EVAL_API_KEY,
    EMB_MODEL, EMB_BASE_URL, EMB_API_KEY, EMB_DIM,
    DATA_NAME, QUESTION_STAGE, CACHES_DIR,
    RAG_METHODS, OUTPUT_DIR,
    RAG_V81_PATH, HYPERRAG_MAIN_DIR, LIGHTRAG_DIR, SIMPLE_RAG_PATH,
)

# ============================================================================
# LLM 调用工具
# ============================================================================

def llm_call(prompt, system_prompt=None, model=None, base_url=None, api_key=None):
    """通用 LLM 调用"""
    client = OpenAI(api_key=api_key or LLM_API_KEY, base_url=base_url or LLM_BASE_URL)
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
        """初始化简单 RAG（基于向量检索）"""
        from openai import OpenAI as _OpenAI
        self.emb_client = _OpenAI(api_key=EMB_API_KEY, base_url=EMB_BASE_URL)
        # 对所有 context 分块并编码
        self.chunks = []
        self.embeddings = []
        chunk_size = 500
        for ctx in self.contexts:
            # 简单分块
            words = ctx.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk.strip():
                    self.chunks.append(chunk)

        if self.chunks:
            print(f"    SimpleRAG: 编码 {len(self.chunks)} 个文本块...")
            # 批量编码（每批 20 个）
            batch_size = 20
            all_embs = []
            for i in range(0, len(self.chunks), batch_size):
                batch = self.chunks[i:i + batch_size]
                try:
                    resp = self.emb_client.embeddings.create(
                        model=EMB_MODEL, input=batch
                    )
                    for item in resp.data:
                        all_embs.append(item.embedding)
                except Exception as e:
                    print(f"    [Warning] Embedding 失败: {e}")
                    all_embs.extend([[0.0] * EMB_DIM] * len(batch))
            self.embeddings = np.array(all_embs)
            # 归一化
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1
            self.embeddings = self.embeddings / norms

    def _get_embedding(self, text):
        try:
            resp = self.emb_client.embeddings.create(model=EMB_MODEL, input=[text])
            emb = np.array(resp.data[0].embedding)
            norm = np.linalg.norm(emb)
            if norm > 0:
                emb = emb / norm
            return emb
        except Exception as e:
            print(f"    [Warning] Query embedding 失败: {e}")
            return np.zeros(EMB_DIM)

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


class HyperRAGv81Adapter(BaseAdapter):
    """HyperRAG v8.1 适配器"""
    name = "hyperrag_v81"

    def __init__(self, contexts=None):
        super().__init__()
        self.contexts = contexts or []
        self.rag_engine = None
        self._init_rag()

    def _init_rag(self):
        """动态加载 rag(8.1).py 并初始化引擎"""
        try:
            spec = importlib.util.spec_from_file_location("rag_v81", RAG_V81_PATH)
            rag_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(rag_module)

            # 获取 HyperRAG 类
            rag_class = getattr(rag_module, "HyperRAG_Attention_v81", None)
            if rag_class is None:
                # 尝试其它可能的类名
                for attr_name in dir(rag_module):
                    attr = getattr(rag_module, attr_name)
                    if isinstance(attr, type) and attr_name.lower().startswith("hyperrag"):
                        rag_class = attr
                        break

            if rag_class:
                self.rag_engine = rag_class(
                    llm_model=LLM_MODEL,
                    llm_base_url=LLM_BASE_URL,
                    llm_api_key=LLM_API_KEY,
                    emb_model=EMB_MODEL,
                    emb_base_url=EMB_BASE_URL,
                    emb_api_key=EMB_API_KEY,
                    emb_dim=EMB_DIM,
                )
                # 构建索引
                if self.contexts:
                    print(f"    HyperRAG v8.1: 索引 {len(self.contexts)} 个文档...")
                    self.rag_engine.build_index(self.contexts)
                print("    HyperRAG v8.1: 初始化完成")
            else:
                print("    [Warning] 未找到 HyperRAG v8.1 类, 将回退为 LLM-only")
        except Exception as e:
            print(f"    [Warning] HyperRAG v8.1 初始化失败: {e}")
            self.rag_engine = None

    def query(self, question, context=None):
        if self.rag_engine is None:
            return llm_call(f"Please answer the following question:\n\n{question}")
        try:
            # 尝试异步调用
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                if hasattr(self.rag_engine, "aquery"):
                    result = loop.run_until_complete(self.rag_engine.aquery(question))
                elif hasattr(self.rag_engine, "query"):
                    result = self.rag_engine.query(question)
                elif hasattr(self.rag_engine, "generate"):
                    result = self.rag_engine.generate(question)
                else:
                    result = llm_call(f"Please answer the following question:\n\n{question}")
            finally:
                loop.close()
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            print(f"    [Warning] HyperRAG v8.1 查询失败: {e}")
            return f"[Error] {e}"


class HyperRAGMainAdapter(BaseAdapter):
    """Hyper-RAG-main 适配器"""
    name = "hyperrag_main"

    def __init__(self, contexts=None):
        super().__init__()
        self.contexts = contexts or []
        self.rag_instance = None
        self._init_rag()

    def _init_rag(self):
        """加载 Hyper-RAG-main 的 HyperRAG 类"""
        try:
            sys.path.insert(0, HYPERRAG_MAIN_DIR)
            from hyperrag import HyperRAG

            working_dir = Path(HYPERRAG_MAIN_DIR) / "caches" / DATA_NAME
            working_dir.mkdir(parents=True, exist_ok=True)

            self.rag_instance = HyperRAG(
                working_dir=str(working_dir),
                llm_model_func=self._llm_func,
                embedding_func={
                    "func": self._emb_func,
                    "embedding_dim": EMB_DIM,
                },
            )

            if self.contexts:
                print(f"    Hyper-RAG-main: 索引 {len(self.contexts)} 个文档...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.rag_instance.ainsert(self.contexts))
                finally:
                    loop.close()
            print("    Hyper-RAG-main: 初始化完成")
        except Exception as e:
            print(f"    [Warning] Hyper-RAG-main 初始化失败: {e}")
            self.rag_instance = None

    async def _llm_func(self, prompt, system_prompt=None, **kwargs):
        return llm_call(prompt, system_prompt)

    async def _emb_func(self, texts):
        client = OpenAI(api_key=EMB_API_KEY, base_url=EMB_BASE_URL)
        resp = client.embeddings.create(model=EMB_MODEL, input=texts)
        return [item.embedding for item in resp.data]

    def query(self, question, context=None):
        if self.rag_instance is None:
            return llm_call(f"Please answer the following question:\n\n{question}")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.rag_instance.aquery(question))
            finally:
                loop.close()
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            print(f"    [Warning] Hyper-RAG-main 查询失败: {e}")
            return f"[Error] {e}"


class LightRAGAdapter(BaseAdapter):
    """LightRAG 适配器"""
    name = "lightrag"

    def __init__(self, contexts=None):
        super().__init__()
        self.contexts = contexts or []
        self.rag_instance = None
        self._init_rag()

    def _init_rag(self):
        """加载 LightRAG"""
        try:
            sys.path.insert(0, LIGHTRAG_DIR)
            from lightrag import LightRAG as _LightRAG, QueryParam

            working_dir = Path(LIGHTRAG_DIR) / "caches" / DATA_NAME
            working_dir.mkdir(parents=True, exist_ok=True)

            self.rag_instance = _LightRAG(
                working_dir=str(working_dir),
                llm_model_func=self._llm_func,
                embedding_func={
                    "func": self._emb_func,
                    "embedding_dim": EMB_DIM,
                },
            )

            if self.contexts:
                print(f"    LightRAG: 索引 {len(self.contexts)} 个文档...")
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    loop.run_until_complete(self.rag_instance.ainsert(self.contexts))
                finally:
                    loop.close()
            print("    LightRAG: 初始化完成")
        except Exception as e:
            print(f"    [Warning] LightRAG 初始化失败: {e}")
            self.rag_instance = None

    async def _llm_func(self, prompt, system_prompt=None, **kwargs):
        return llm_call(prompt, system_prompt)

    async def _emb_func(self, texts):
        client = OpenAI(api_key=EMB_API_KEY, base_url=EMB_BASE_URL)
        resp = client.embeddings.create(model=EMB_MODEL, input=texts)
        return [item.embedding for item in resp.data]

    def query(self, question, context=None):
        if self.rag_instance is None:
            return llm_call(f"Please answer the following question:\n\n{question}")
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                result = loop.run_until_complete(self.rag_instance.aquery(question))
            finally:
                loop.close()
            return result if isinstance(result, str) else str(result)
        except Exception as e:
            print(f"    [Warning] LightRAG 查询失败: {e}")
            return f"[Error] {e}"


# ============================================================================
# 评分系统（复用 step_4_evaluate_scoring.py 的逻辑）
# ============================================================================

SCORING_SYSTEM_PROMPT = """---Role---
You are an expert tasked with evaluating answers to the questions by using the relevant documents based on five criteria:**Comprehensiveness**, **Diversity**,**Empowerment**, **Logical**,and **Readability** ."""

SCORING_PROMPT_TEMPLATE = """You will evaluate the answers to the questions by using the relevant documents based on five criteria:**Comprehensiveness**, **Diversity**,**Empowerment**, **Logical**,and **Readability** .

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
    }}
}}"""


def score_single_answer(query, answer, reference):
    """对单个回答进行 5 维度评分"""
    prompt = SCORING_PROMPT_TEMPLATE.format(
        reference=reference, query=query, answer=answer
    )
    try:
        response = eval_llm_call(prompt, SCORING_SYSTEM_PROMPT)
        return response
    except Exception as e:
        print(f"    [Error] 评分失败: {e}")
        return "{}"


def parse_scores(response_text):
    """从评分响应中解析 5 个维度分数"""
    metrics = ["Comprehensiveness", "Diversity", "Empowerment", "Logical", "Readability"]
    scores = {}
    try:
        score_values = re.findall(r'"Score":\s*"?(\d+)"?', response_text)
        if len(score_values) >= 5:
            for i, metric in enumerate(metrics):
                scores[metric] = float(score_values[i])
    except Exception:
        pass
    return scores


def compute_aggregate_scores(all_scores):
    """计算所有问题的平均分"""
    metrics = ["Comprehensiveness", "Diversity", "Empowerment", "Logical", "Readability"]
    totals = {m: 0.0 for m in metrics}
    valid_count = 0

    for scores in all_scores:
        if len(scores) == 5:
            for m in metrics:
                totals[m] += scores.get(m, 0)
            valid_count += 1

    if valid_count == 0:
        return {m: 0.0 for m in metrics + ["Average"]}

    result = {m: totals[m] / valid_count for m in metrics}
    result["Average"] = sum(result.values()) / len(metrics)
    return result


# ============================================================================
# 主实验流程
# ============================================================================

def create_adapter(method_name, contexts):
    """根据方法名创建对应的适配器"""
    adapters = {
        "llm_only": LLMOnlyAdapter,
        "simple_rag": lambda: SimpleRAGAdapter(contexts),
        "hyperrag_v81": lambda: HyperRAGv81Adapter(contexts),
        "hyperrag_main": lambda: HyperRAGMainAdapter(contexts),
        "lightrag": lambda: LightRAGAdapter(contexts),
    }
    factory = adapters.get(method_name)
    if factory is None:
        print(f"  [Error] 未知的 RAG 方法: {method_name}")
        return None
    if callable(factory) and not isinstance(factory, type):
        return factory()
    return factory()


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
    contexts_file = Path(CACHES_DIR) / DATA_NAME / "contexts" / f"{DATA_NAME}_unique_contexts.json"
    contexts = []
    if contexts_file.exists():
        with open(contexts_file, "r", encoding="utf-8") as f:
            contexts = json.load(f)
        print(f"  已加载 {len(contexts)} 个 context 文档")
    else:
        print(f"  [Warning] Context 文件不存在: {contexts_file}")

    # 3. 准备输出目录
    output_root = Path(OUTPUT_DIR) / LLM_MODEL / DATA_NAME
    output_root.mkdir(parents=True, exist_ok=True)

    # 4. 对每个 RAG 方法运行实验
    all_results = {}

    for method_name in RAG_METHODS:
        print(f"\n{'='*60}")
        print(f"[Step 3] 运行 RAG 方法: {method_name}")
        print(f"{'='*60}")

        # 检查是否已有缓存的回答结果
        response_file = output_root / f"{method_name}_{QUESTION_STAGE}_stage_result.json"
        scoring_file = output_root / f"{method_name}_{QUESTION_STAGE}_stage_scoring.json"

        # 3a. 生成回答
        if response_file.exists():
            print(f"  已有缓存结果，跳过回答生成: {response_file}")
            with open(response_file, "r", encoding="utf-8") as f:
                answers_data = json.load(f)
            answers = [item["result"] for item in answers_data]
        else:
            print(f"  初始化 {method_name}...")
            adapter = create_adapter(method_name, contexts)
            if adapter is None:
                continue

            answers = []
            answers_data = []
            for i, question in enumerate(questions):
                print(f"  [{i+1}/{len(questions)}] {question[:60]}...")
                try:
                    answer = adapter.query(question)
                    answers.append(answer)
                    answers_data.append({
                        "query": question,
                        "result": answer,
                    })
                except Exception as e:
                    print(f"    [Error] 回答生成失败: {e}")
                    answers.append(f"[Error] {e}")
                    answers_data.append({
                        "query": question,
                        "result": f"[Error] {e}",
                    })

            # 保存回答
            with open(response_file, "w", encoding="utf-8") as f:
                json.dump(answers_data, f, indent=2, ensure_ascii=False)
            print(f"  回答已保存: {response_file}")

            adapter.cleanup()

        # 3b. 评分
        if scoring_file.exists():
            print(f"  已有缓存评分，跳过评分: {scoring_file}")
            with open(scoring_file, "r", encoding="utf-8") as f:
                scoring_data = json.load(f)
            all_parsed_scores = [parse_scores(s) for s in scoring_data["raw_responses"]]
        else:
            print(f"\n  开始评分 ({method_name})...")
            raw_responses = []
            all_parsed_scores = []
            for i, (q, a, r) in enumerate(zip(questions, answers, refs)):
                print(f"  评分 [{i+1}/{len(questions)}]...")
                resp = score_single_answer(q, a, r)
                raw_responses.append(resp)
                scores = parse_scores(resp)
                all_parsed_scores.append(scores)

            # 保存评分详情
            scoring_data = {
                "method": method_name,
                "model": LLM_MODEL,
                "eval_model": EVAL_MODEL,
                "raw_responses": raw_responses,
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

    # 5. 输出总汇表
    print(f"\n{'='*70}")
    print(f"  实验结果汇总  (LLM: {LLM_MODEL}, Dataset: {DATA_NAME}, Stage: {QUESTION_STAGE})")
    print(f"{'='*70}")

    metrics = ["Comprehensiveness", "Diversity", "Empowerment", "Logical", "Readability", "Average"]

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

    # 保存汇总到 JSON
    summary = {
        "llm_model": LLM_MODEL,
        "eval_model": EVAL_MODEL,
        "dataset": DATA_NAME,
        "question_stage": QUESTION_STAGE,
        "timestamp": datetime.now().isoformat(),
        "results": all_results,
    }
    summary_file = output_root / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n汇总已保存: {summary_file}")

    return all_results


if __name__ == "__main__":
    run_experiment()
