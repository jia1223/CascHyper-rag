# -*- coding: utf-8 -*-
"""
Compare two HyperRAG v8.1 query-scoring formulas on the cached duibi index.

This script does not rebuild the index. It loads:
  - duibi/hyperrag_v81_gpt-4o-mini_6105367703bc2c1e.pkl
  - duibi/data/caches_v81/mechanical/questions/{stage}_stage.json
  - duibi/data/caches_v81/mechanical/questions/{stage}_stage_ref.json

Compared methods:
  1. shared_exp_query:
     s_c = q_exp dot h_c(final), s_s = q_exp dot h_s(final), s_e = Entity MaxSim
  2. level_specific_query:
     s_c = q_c dot h_c(final), s_s = q_s dot h_s(final), s_e = Entity MaxSim

Both methods use the same cached index, same query expansion, same entity extraction,
same coarse chunk candidates, same answer-generation prompt, and the same six-metric
LLM scoring rubric used by run_experiment copy 3.py.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
import pickle
import re
import sys
import time
import types
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
from openai import OpenAI


BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
CACHE_PATH = BASE_DIR / "hyperrag_v81_gpt-4o-mini_6105367703bc2c1e.pkl"
CACHES_DIR = BASE_DIR / "data" / "caches_v81"
OUTPUT_DIR = BASE_DIR / "results"
QUERY_CACHE_DIR = BASE_DIR / "query_cache"

DATA_NAME = "mechanical"
DEFAULT_STAGE = 1

LLM_MODEL = "gpt-4o-mini"
EVAL_MODEL = "gpt-4o-mini"
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.openai-proxy.org/v1") 
EVAL_BASE_URL = os.getenv("EVAL_BASE_URL", LLM_BASE_URL)
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
EVAL_API_KEY = os.getenv("EVAL_API_KEY", LLM_API_KEY)

EMB_MODEL = os.getenv("EMB_MODEL", "BAAI/bge-m3")
EMB_BASE_URL = os.getenv("EMB_BASE_URL", "https://api.siliconflow.cn/v1")
EMB_API_KEY = os.getenv("EMB_API_KEY", "")

TOP_K_CHUNKS = 10
TOP_K_SENTS = 10
TEMPERATURE_FUSION = 0.1
TEMPERATURE_SCORING = 0.08

CORE_METRICS = ["Comprehensiveness", "Diversity", "Empowerment", "Logical", "Readability"]
RELEVANCE_METRIC = "Relevance"
ALL_METRICS = CORE_METRICS + [RELEVANCE_METRIC]


QUERY_EXPAND_PROMPT = """
You are a search query optimizer.
Expand the user's query into a more detailed question containing relevant keywords,
context, and possible entities.

Rules:
1. If the original query is a question, the expanded query must remain a question.
2. Do not answer the question. Only rewrite and expand the question itself.
3. Preserve the user's core intent.

User query: {query}
Return only the expanded question, with no explanation:
"""

ENTITY_EXTRACTION_PROMPT = """
Extract key entities from the sentence or question. Return only valid JSON.
Each entity should have name, type, importance, and description.

Text:
{sentence}

JSON format:
{{"entities":[{{"name":"...","type":"...","importance":1.0,"description":"..."}}]}}
"""

ANSWER_SYSTEM_PROMPT = """You are a careful technical QA assistant. Answer the question using only the supplied retrieved evidence. If the evidence is insufficient, say that the material is insufficient. Keep the answer focused, factual, and readable."""

ANSWER_PROMPT_TEMPLATE = """
Question:
{query}

Retrieved evidence:
{context}

Generate the final answer:
"""

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
    "Comprehensiveness": {{"Explanation": "Provide explanation here", "Level": "A single number 1-5", "Score": "A single number 0-100"}},
    "Diversity": {{"Explanation": "Provide explanation here", "Level": "A single number 1-5", "Score": "A single number 0-100"}},
    "Empowerment": {{"Explanation": "Provide explanation here", "Level": "A single number 1-5", "Score": "A single number 0-100"}},
    "Logical": {{"Explanation": "Provide explanation here", "Level": "A single number 1-5", "Score": "A single number 0-100"}},
    "Readability": {{"Explanation": "Provide explanation here", "Level": "A single number 1-5", "Score": "A single number 0-100"}},
    "Relevance": {{"Explanation": "Provide explanation here", "Level": "A single number 1-5", "Score": "A single number 0-100"}}
}}"""


@dataclass
class EntityLite:
    name: str
    weight: float = 1.0
    raw_embedding: Optional[np.ndarray] = None
    final_embedding: Optional[np.ndarray] = None


def install_pickle_classes() -> None:
    """Allow loading pickle objects whose classes were saved under module rag_v81."""
    if "rag_v81" in sys.modules:
        return
    module = types.ModuleType("rag_v81")
    sys.modules["rag_v81"] = module
    for name in ("EntityNode", "SentenceNode", "ChunkNode", "TopicNode"):
        cls = type(name, (), {"__module__": "rag_v81"})
        setattr(module, name, cls)


def normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 1e-9 else vec


def softmax(values: Iterable[float], temperature: float) -> np.ndarray:
    logits = np.array(list(values), dtype=float) / temperature
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / (np.sum(exp_logits) + 1e-9)


def attention_fusion(query_vec: np.ndarray, key_vecs: List[np.ndarray], temperature: float = TEMPERATURE_FUSION) -> np.ndarray:
    if not key_vecs:
        return query_vec
    sims = [float(np.dot(query_vec, key)) for key in key_vecs]
    weights = softmax(sims, temperature)
    fused = np.zeros_like(query_vec)
    for weight, key in zip(weights, key_vecs):
        fused += weight * key
    return normalize(fused)


def dynamic_score(sim_c: float, sim_s: float, sim_e: float, temperature: float = TEMPERATURE_SCORING) -> Tuple[float, Dict[str, float]]:
    weights = softmax([sim_c, sim_s, sim_e], temperature)
    score = float(np.sum(weights * np.array([sim_c, sim_s, sim_e], dtype=float)))
    return score, {"w_c": float(weights[0]), "w_s": float(weights[1]), "w_e": float(weights[2])}


def make_client(api_key: Optional[str], base_url: str) -> OpenAI:
    if not api_key:
        raise RuntimeError("Missing API key. Set OPENAI_API_KEY/LLM_API_KEY and EMB_API_KEY/SILICONFLOW_API_KEY if needed.")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=120, max_retries=3)


def llm_call(prompt: str, system_prompt: Optional[str] = None, *, model: str = LLM_MODEL, base_url: str = LLM_BASE_URL, api_key: Optional[str] = LLM_API_KEY) -> str:
    client = make_client(api_key, base_url)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def eval_llm_call(prompt: str, system_prompt: Optional[str] = None) -> str:
    return llm_call(prompt, system_prompt, model=EVAL_MODEL, base_url=EVAL_BASE_URL, api_key=EVAL_API_KEY)


def embedding_call(texts: List[str]) -> List[np.ndarray]:
    if not texts:
        return []
    client = make_client(EMB_API_KEY, EMB_BASE_URL)
    all_embeddings: List[np.ndarray] = []
    for start in range(0, len(texts), 50):
        batch = texts[start : start + 50]
        response = client.embeddings.create(model=EMB_MODEL, input=batch)
        all_embeddings.extend(np.array(item.embedding, dtype=float) for item in response.data)
    return all_embeddings


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_index() -> Dict[str, Any]:
    install_pickle_classes()
    with CACHE_PATH.open("rb") as f:
        return NumpyCompatUnpickler(f).load()


class NumpyCompatUnpickler(pickle.Unpickler):
    """Read NumPy 2.x pickles in older NumPy environments when possible."""

    def find_class(self, module: str, name: str):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)


def cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:24]


def extract_json_payload(text: str) -> Optional[dict]:
    candidates = re.findall(r"```(?:json)?\s*(.*?)```", text or "", flags=re.DOTALL)
    candidates.append(text or "")
    for candidate in candidates:
        start, end = candidate.find("{"), candidate.rfind("}")
        if start < 0 or end <= start:
            continue
        try:
            payload = json.loads(candidate[start : end + 1])
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            return payload
    return None


def extract_entities(text: str) -> List[EntityLite]:
    prompt = ENTITY_EXTRACTION_PROMPT.format(sentence=text)
    response = llm_call(prompt)
    payload = extract_json_payload(response) or {}
    entities = []
    for item in payload.get("entities", []) or []:
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        try:
            weight = float(item.get("importance", 1.0))
        except (TypeError, ValueError):
            weight = 1.0
        entities.append(EntityLite(name=name, weight=max(weight, 0.0)))
    return entities


def load_or_create_query_state(query: str, index: Dict[str, Any]) -> Dict[str, Any]:
    QUERY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = QUERY_CACHE_DIR / f"{cache_key(query)}.pkl"
    if path.exists():
        with path.open("rb") as f:
            return pickle.load(f)

    expanded = llm_call(QUERY_EXPAND_PROMPT.format(query=query))
    q_raw = embedding_call([expanded])[0]

    q_entities = extract_entities(expanded)
    if q_entities:
        ent_vecs = embedding_call([e.name for e in q_entities])
        for ent, vec in zip(q_entities, ent_vecs):
            ent.raw_embedding = vec
        weights = np.array([e.weight for e in q_entities], dtype=float).reshape(-1, 1)
        q_ent_agg = normalize(np.sum(np.stack(ent_vecs) * weights, axis=0) / (np.sum(weights) + 1e-9))
    else:
        q_ent_agg = q_raw

    topic_scores = [(tid, float(np.dot(q_raw, topic.topic_vector))) for tid, topic in index["topics"].items()]
    topic_scores.sort(key=lambda item: item[1], reverse=True)
    selected_topics = []
    if topic_scores:
        max_score = topic_scores[0][1]
        for tid, score in topic_scores:
            if score >= max_score * 0.75 and score > 0.3:
                selected_topics.append((tid, score))
            else:
                break

    total_w = sum(score for _, score in selected_topics)
    if total_w > 0:
        q_topic = np.zeros_like(q_raw)
        for tid, score in selected_topics:
            q_topic += score * index["topics"][tid].topic_vector
        q_topic = normalize(q_topic / (total_w + 1e-9))
    else:
        q_topic = q_raw

    q_chunk = attention_fusion(q_raw, [q_topic, q_raw])
    q_sent = attention_fusion(q_raw, [q_chunk, q_ent_agg, q_raw])

    q_ent_final = []
    for ent in q_entities:
        ent.final_embedding = attention_fusion(ent.raw_embedding, [q_sent, ent.raw_embedding])
        q_ent_final.append(ent.final_embedding)

    state = {
        "query": query,
        "expanded": expanded,
        "entities": [e.name for e in q_entities],
        "q_raw": q_raw,
        "q_chunk": q_chunk,
        "q_sent": q_sent,
        "q_ent_final": q_ent_final,
        "selected_tids": {tid for tid, _ in selected_topics},
    }
    with path.open("wb") as f:
        pickle.dump(state, f)
    return state


def entity_maxsim(q_ent_final: List[np.ndarray], sent: Any) -> float:
    if not q_ent_final or not getattr(sent, "entities", None):
        return 0.0
    doc_vecs = [ent.final_embedding for ent in sent.entities if getattr(ent, "final_embedding", None) is not None]
    if not doc_vecs:
        return 0.0
    sims = np.dot(np.stack(q_ent_final), np.stack(doc_vecs).T)
    return float(np.mean(np.max(sims, axis=1)))


def retrieve(index: Dict[str, Any], q_state: Dict[str, Any], method: str) -> Dict[str, Any]:
    chunks = index["chunks"]
    sentences = index["sentences"]
    selected_tids = q_state["selected_tids"]

    chunk_candidates = []
    for cid, chunk in chunks.items():
        chunk_topics = set(getattr(chunk, "topic_memberships", {}).keys())
        if not chunk_topics.isdisjoint(selected_tids) or not selected_tids:
            score = float(np.dot(q_state["q_chunk"], chunk.final_embedding))
            chunk_candidates.append((cid, score))
    chunk_candidates.sort(key=lambda item: item[1], reverse=True)
    top_chunks = chunk_candidates[:TOP_K_CHUNKS]

    rows = []
    for cid, chunk_score in top_chunks:
        chunk = chunks[cid]
        for sid in getattr(chunk, "sentence_ids", []):
            sent = sentences[sid]
            sim_e = entity_maxsim(q_state["q_ent_final"], sent)
            if method == "shared_exp_query":
                sim_c = float(np.dot(q_state["q_raw"], chunk.final_embedding))
                sim_s = float(np.dot(q_state["q_raw"], sent.final_embedding))
            elif method == "level_specific_query":
                sim_c = float(np.dot(q_state["q_chunk"], chunk.final_embedding))
                sim_s = float(np.dot(q_state["q_sent"], sent.final_embedding))
            else:
                raise ValueError(f"Unknown method: {method}")
            score, weights = dynamic_score(sim_c, sim_s, sim_e)
            rows.append({
                "sent_id": sid,
                "chunk_id": cid,
                "text": sent.text,
                "score": score,
                "sim_c": sim_c,
                "sim_s": sim_s,
                "sim_e": sim_e,
                "attn_weights": weights,
            })

    rows.sort(key=lambda item: item["score"], reverse=True)
    return {
        "hop1": rows[:TOP_K_SENTS],
        "top_chunks": [{"chunk_id": cid, "score": score, "text": chunks[cid].text} for cid, score in top_chunks],
    }


def build_answer_context(search_result: Dict[str, Any]) -> str:
    pieces = []
    for i, item in enumerate(search_result.get("hop1", []), 1):
        pieces.append(f"[Evidence {i}] score={item['score']:.4f}\n{item['text']}")
    return "\n\n".join(pieces)


def generate_answer(query: str, search_result: Dict[str, Any]) -> str:
    prompt = ANSWER_PROMPT_TEMPLATE.format(query=query, context=build_answer_context(search_result))
    return llm_call(prompt, ANSWER_SYSTEM_PROMPT)


def _parse_metric_score(response_text: str, metric: str) -> Optional[float]:
    payload = extract_json_payload(response_text)
    if payload and isinstance(payload.get(metric), dict):
        try:
            return float(payload[metric].get("Score"))
        except (TypeError, ValueError):
            pass
    pattern = rf'"{re.escape(metric)}"\s*:\s*\{{.*?"Score"\s*:\s*"?(\d+(?:\.\d+)?)"?'
    match = re.search(pattern, response_text or "", flags=re.DOTALL)
    return float(match.group(1)) if match else None


def parse_scores(response_text: str, metrics: List[str] = ALL_METRICS) -> Dict[str, float]:
    scores = {}
    for metric in metrics:
        value = _parse_metric_score(response_text, metric)
        if value is not None:
            scores[metric] = value
    return scores


def compute_aggregate_scores(rows: List[Dict[str, float]]) -> Dict[str, float]:
    valid_rows = [row for row in rows if all(metric in row for metric in ALL_METRICS)]
    if not valid_rows:
        return {metric: 0.0 for metric in ALL_METRICS + ["Average"]}
    result = {metric: float(np.mean([row[metric] for row in valid_rows])) for metric in ALL_METRICS}
    result["Average"] = float(np.mean([result[metric] for metric in ALL_METRICS]))
    return result


def score_answer(query: str, answer: str, reference: str) -> str:
    prompt = SCORING_PROMPT_TEMPLATE.format(reference=reference, query=query, answer=answer)
    return eval_llm_call(prompt, SCORING_SYSTEM_PROMPT)


def load_questions_and_refs(stage: int) -> Tuple[List[str], List[str]]:
    question_path = CACHES_DIR / DATA_NAME / "questions" / f"{stage}_stage.json"
    ref_path = CACHES_DIR / DATA_NAME / "questions" / f"{stage}_stage_ref.json"
    questions = load_json(question_path)
    refs = load_json(ref_path)
    if len(questions) != len(refs):
        raise RuntimeError(f"Question/ref count mismatch: {len(questions)} vs {len(refs)}")
    return questions, refs


def retry(label: str, func, max_attempts: int = 3):
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            return func()
        except Exception as exc:
            last_error = exc
            if attempt == max_attempts:
                break
            delay = min(30, 5 * (2 ** (attempt - 1)))
            print(f"    {label} failed ({attempt}/{max_attempts}): {exc}; retry in {delay}s")
            time.sleep(delay)
    raise RuntimeError(f"{label} failed after {max_attempts} attempts: {last_error}") from last_error


def run(stage: int, limit: Optional[int] = None, skip_generation: bool = False, skip_scoring: bool = False) -> Dict[str, Any]:
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    questions, refs = load_questions_and_refs(stage)
    if limit is not None:
        questions = questions[:limit]
        refs = refs[:limit]

    print(f"Loading cached index: {CACHE_PATH}")
    index = load_index()
    print(f"Index stats: chunks={len(index['chunks'])}, sentences={len(index['sentences'])}, topics={len(index['topics'])}")

    output_root = OUTPUT_DIR / LLM_MODEL / DATA_NAME
    output_root.mkdir(parents=True, exist_ok=True)
    methods = ["shared_exp_query", "level_specific_query"]
    all_results: Dict[str, Dict[str, float]] = {}

    for method in methods:
        print(f"\n{'=' * 70}\nMethod: {method}\n{'=' * 70}")
        result_path = output_root / f"{method}_{stage}_stage_result.json"
        retrieval_path = output_root / f"{method}_{stage}_stage_retrieval.json"
        scoring_path = output_root / f"{method}_{stage}_stage_scoring.json"

        if result_path.exists():
            answers_data = load_json(result_path)
            print(f"Loaded cached answers: {result_path}")
        else:
            answers_data = []

        if retrieval_path.exists():
            retrieval_data = load_json(retrieval_path)
        else:
            retrieval_data = []

        while len(answers_data) < len(questions):
            idx = len(answers_data)
            query = questions[idx]
            print(f"  Answer [{idx + 1}/{len(questions)}] using existing question: {query[:80]}...")
            q_state = retry("query state", lambda: load_or_create_query_state(query, index))
            search_result = retrieve(index, q_state, method)
            retrieval_data.append({
                "query": query,
                "expanded_query": q_state["expanded"],
                "entities": q_state["entities"],
                "results": search_result,
            })
            if skip_generation:
                answer = build_answer_context(search_result)
            else:
                answer = retry("answer generation", lambda: generate_answer(query, search_result))
            answers_data.append({"query": query, "result": answer})
            save_json(result_path, answers_data)
            save_json(retrieval_path, retrieval_data)

        if skip_scoring:
            all_results[method] = {}
            continue

        if scoring_path.exists():
            scoring_data = load_json(scoring_path)
            raw_responses = scoring_data.get("raw_responses", [])
            print(f"Loaded cached scoring: {scoring_path}")
        else:
            raw_responses = []

        while len(raw_responses) < len(questions):
            idx = len(raw_responses)
            print(f"  Score [{idx + 1}/{len(questions)}]...")
            answer = answers_data[idx]["result"]
            response = retry("LLM scoring", lambda: score_answer(questions[idx], answer, refs[idx]))
            raw_responses.append(response)
            scoring_data = {
                "method": method,
                "model": LLM_MODEL,
                "eval_model": EVAL_MODEL,
                "dataset": DATA_NAME,
                "question_stage": stage,
                "metrics": ALL_METRICS,
                "raw_responses": raw_responses,
            }
            save_json(scoring_path, scoring_data)

        score_rows = [parse_scores(resp) for resp in raw_responses]
        aggregate = compute_aggregate_scores(score_rows)
        all_results[method] = aggregate
        print(f"Aggregate: {aggregate}")

    winner = None
    if all_results and all(score for score in all_results.values()):
        winner = max(all_results, key=lambda name: all_results[name].get("Average", 0.0))

    summary = {
        "llm_model": LLM_MODEL,
        "eval_model": EVAL_MODEL,
        "dataset": DATA_NAME,
        "question_stage": stage,
        "timestamp": datetime.now().isoformat(),
        "compared_methods": {
            "shared_exp_query": "s_c=q_exp dot h_c, s_s=q_exp dot h_s, s_e=entity MaxSim",
            "level_specific_query": "s_c=q_c dot h_c, s_s=q_s dot h_s, s_e=entity MaxSim",
        },
        "results": all_results,
        "winner_by_average": winner,
    }
    summary_path = output_root / f"summary_query_scoring_{stage}_stage.json"
    save_json(summary_path, summary)
    print(f"\nSummary saved: {summary_path}")
    if winner:
        print(f"Winner by Average: {winner} ({all_results[winner]['Average']:.2f})")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two query-scoring formulas on cached HyperRAG v8.1 index.")
    parser.add_argument("--stage", type=int, default=DEFAULT_STAGE, choices=[1, 2, 3])
    parser.add_argument("--limit", type=int, default=None, help="Optional question limit for a smoke test.")
    parser.add_argument("--skip-generation", action="store_true", help="Use retrieved evidence as the answer; useful for retrieval-only smoke tests.")
    parser.add_argument("--skip-scoring", action="store_true", help="Generate answers only, without LLM scoring.")
    args = parser.parse_args()
    run(args.stage, limit=args.limit, skip_generation=args.skip_generation, skip_scoring=args.skip_scoring)


if __name__ == "__main__":
    main()
