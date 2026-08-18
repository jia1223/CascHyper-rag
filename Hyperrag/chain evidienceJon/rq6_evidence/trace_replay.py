"""Replay existing Physics indexes and export canonical RQ6 retrieval traces.

This module is deliberately an adapter: it never changes either RAG implementation
or either persisted index.  It exposes the retrieval records that those
implementations already select, then maps them to the frozen original-document
spans required by the RQ6 evaluator.
"""

from __future__ import annotations

import asyncio
import bisect
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import os
import re
import sys
from array import array
from pathlib import Path
from typing import Any

from .io_utils import read_json, sha256_file, write_json
from .matched_final_protocol import (
    MATCHED_FINAL_CONTEXT_PROTOCOL,
    MATCHED_FINAL_SOURCE_TOKEN_BUDGET,
    select_complete_source_units,
)
from .native_protocol import NATIVE_EVIDENCE_PROTOCOL, native_retrieval_config
from .validation import (
    validate_final_context_traces,
    validate_latent_topic_manifest,
    validate_matched_final_context_traces,
    validate_native_evidence_traces,
    validate_traces,
)


class ProvenanceError(ValueError):
    """A retrieved unit cannot be mapped unambiguously to the frozen corpus."""


def _normalized(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


class CanonicalSentenceMapper:
    """Map index text to canonical spans using its original-document provenance."""

    def __init__(self, manifest: dict[str, Any], contexts: list[str]) -> None:
        document_hashes = {
            item["source_sha256"]: item["document_id"] for item in manifest["documents"]
        }
        self.document_texts: dict[str, str] = {}
        for text in contexts:
            document_id = document_hashes.get(hashlib.sha256(text.encode("utf-8")).hexdigest())
            if document_id is None:
                raise ProvenanceError("A source context is not represented by the frozen corpus manifest")
            self.document_texts[document_id] = text
        if len(self.document_texts) != len(document_hashes):
            raise ProvenanceError("The source contexts do not cover every frozen corpus document")
        self.sentences_by_document: dict[str, list[dict[str, Any]]] = {}
        self.sentence_by_id: dict[str, dict[str, Any]] = {}
        for sentence in manifest["sentences"]:
            self.sentences_by_document.setdefault(sentence["document_id"], []).append(sentence)
            self.sentence_by_id[str(sentence["sentence_id"])] = sentence
        self.combined_text = "\n\n".join(contexts)
        self.normalized_combined, self.normalized_positions = self._normalized_with_positions(self.combined_text)
        self.document_ranges: dict[str, tuple[int, int]] = {}
        cursor = 0
        for text in contexts:
            document_id = document_hashes[hashlib.sha256(text.encode("utf-8")).hexdigest()]
            self.document_ranges[document_id] = (cursor, cursor + len(text))
            cursor += len(text) + 2
        self.source_spans = sorted(
            (
                self.document_ranges[document_id][0] + int(item["char_start"]),
                self.document_ranges[document_id][0] + int(item["char_end"]),
                str(item["sentence_id"]),
            )
            for document_id, items in self.sentences_by_document.items()
            for item in items
        )
        self.source_starts = [item[0] for item in self.source_spans]

    @staticmethod
    def _unique_occurrence(haystack: str, needle: str, label: str) -> int:
        start = haystack.find(needle)
        if start < 0:
            raise ProvenanceError(f"{label} is absent from its original document")
        if haystack.find(needle, start + 1) >= 0:
            raise ProvenanceError(f"{label} occurs more than once; source provenance is ambiguous")
        return start

    @staticmethod
    def _normalized_with_positions(text: str) -> tuple[str, array]:
        characters: list[str] = []
        positions = array("I")
        for position, character in enumerate(text.casefold()):
            if "a" <= character <= "z" or "0" <= character <= "9":
                characters.append(character)
                positions.append(position)
        return "".join(characters), positions

    def document_for_text(self, text: str) -> tuple[str, tuple[int, int]]:
        matches: list[tuple[str, int]] = []
        for document_id, document_text in self.document_texts.items():
            start = document_text.find(text)
            if start >= 0:
                if document_text.find(text, start + 1) >= 0:
                    raise ProvenanceError("Retrieved chunk repeats within a document; cannot disambiguate it")
                matches.append((document_id, start))
        if len(matches) != 1:
            raise ProvenanceError(f"Retrieved chunk maps to {len(matches)} source documents, not exactly one")
        document_id, start = matches[0]
        return document_id, (start, start + len(text))

    def chunk_sentence_ids(self, document_id: str, text: str) -> list[str]:
        start = self._unique_occurrence(self.document_texts[document_id], text, "Retrieved chunk")
        return self._sentence_ids_for_document_span(document_id, start, start + len(text))

    def chunk_sentence_ids_at_occurrence(
        self, document_id: str, text: str, occurrence_index: int, expected_occurrences: int,
    ) -> list[str]:
        """Map a baseline chunk using its stable occurrence among identical cached chunks."""
        starts: list[int] = []
        start = self.document_texts[document_id].find(text)
        while start >= 0:
            starts.append(start)
            start = self.document_texts[document_id].find(text, start + 1)
        if len(starts) != expected_occurrences or not 0 <= occurrence_index < len(starts):
            raise ProvenanceError("Hyper-RAG duplicate chunk occurrence cannot be reconciled with the frozen corpus")
        start = starts[occurrence_index]
        return self._sentence_ids_for_document_span(document_id, start, start + len(text))

    def _sentence_ids_for_document_span(self, document_id: str, start: int, end: int) -> list[str]:
        source_ids = [
            item["sentence_id"]
            for item in self.sentences_by_document[document_id]
            if int(item["char_end"]) > start and int(item["char_start"]) < end
        ]
        if not source_ids:
            raise ProvenanceError("Retrieved chunk overlaps no canonical sentence")
        return source_ids

    def engine_chunk_span(self, text: str) -> tuple[int, int]:
        start = self._unique_occurrence(self.combined_text, text, "CascHyper-RAG chunk")
        return start, start + len(text)

    def engine_chunk_span_from_normalized_position(self, normalized_start: int, normalized_length: int) -> tuple[int, int]:
        if normalized_start < 0 or normalized_start + normalized_length > len(self.normalized_positions):
            raise ProvenanceError("CascHyper-RAG normalized chunk span is outside the source corpus")
        return self.normalized_positions[normalized_start], self.normalized_positions[normalized_start + normalized_length - 1] + 1

    def engine_chunk_sentence_ids(self, text: str) -> list[str]:
        start, end = self.engine_chunk_span(text)
        return self.source_sentence_ids_for_span(start, end)

    def source_sentence_ids_for_span(self, start: int, end: int) -> list[str]:
        first = max(0, bisect.bisect_left(self.source_starts, start) - 1)
        while first > 0 and self.source_spans[first - 1][1] > start:
            first -= 1
        # The source spans are ordered by start; the first later span ends the scan.
        source_ids = []
        for left, right, sentence_id in self.source_spans[first:]:
            if left >= end:
                break
            if right > start:
                source_ids.append(sentence_id)
        if not source_ids:
            raise ProvenanceError("CascHyper-RAG chunk overlaps no canonical sentence")
        return source_ids

    def engine_sentence_ids(self, parent_start: int, parent_text: str, retrieved_text: str) -> list[str]:
        relative_start = parent_text.find(retrieved_text)
        if relative_start >= 0:
            start, end = parent_start + relative_start, parent_start + relative_start + len(retrieved_text)
            matches = self.source_sentence_ids_for_span(start, end)
            if matches:
                return matches
        parent_end = parent_start + len(parent_text)
        parent_documents = {
            document_id for document_id, (left, right) in self.document_ranges.items() if right > parent_start and left < parent_end
        }
        normalized = _normalized(retrieved_text)
        candidates = [
            str(item["sentence_id"])
            for document_id in parent_documents
            for item in self.sentences_by_document[document_id]
            if len(_normalized(item["text"])) >= 40
            and (normalized == _normalized(item["text"]) or normalized in _normalized(item["text"]) or _normalized(item["text"]) in normalized)
        ]
        if candidates:
            # v8.1 semantic extraction can return one long unit spanning several
            # consecutive frozen sentences.  Preserve every directly covered span
            # rather than arbitrarily selecting one of them.
            return candidates
        raise ProvenanceError("CascHyper-RAG sentence has no canonical candidate in its parent source span")

    def sentence_ids(self, document_id: str, parent_text: str, retrieved_text: str) -> list[str]:
        """Map one engine sentence within a known source chunk.

        Exact character spans are mandatory when available.  A normalized fallback
        is allowed only when it identifies one canonical sentence in the same
        document; it never silently chooses among duplicate text.
        """
        parent_start = self._unique_occurrence(self.document_texts[document_id], parent_text, "Parent chunk")
        relative_start = parent_text.find(retrieved_text)
        if relative_start >= 0:
            start, end = parent_start + relative_start, parent_start + relative_start + len(retrieved_text)
            matches = [
                item["sentence_id"]
                for item in self.sentences_by_document[document_id]
                if int(item["char_end"]) > start and int(item["char_start"]) < end
            ]
            if matches:
                return matches
        normalized = _normalized(retrieved_text)
        candidates = [
            item["sentence_id"]
            for item in self.sentences_by_document[document_id]
            if len(_normalized(item["text"])) >= 40
            and (normalized == _normalized(item["text"]) or normalized in _normalized(item["text"]) or _normalized(item["text"]) in normalized)
        ]
        if len(candidates) != 1:
            raise ProvenanceError(
                f"Retrieved sentence has {len(candidates)} canonical candidates in {document_id}; refusing an ambiguous mapping"
            )
        return candidates


def _remove_hyperrag_modules() -> None:
    for name in list(sys.modules):
        if name == "hyperrag" or name.startswith("hyperrag."):
            del sys.modules[name]


def _tree_digest(path: Path) -> str:
    """Stable digest used to assert the two persisted retrieval indexes stay unchanged."""
    path = Path(path)
    digest = hashlib.sha256()
    files = [path] if path.is_file() else sorted(item for item in path.rglob("*") if item.is_file() and item.name != "HyperRAG.log")
    for item in files:
        digest.update(str(item.relative_to(path.parent)).encode("utf-8"))
        with item.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _question_items(split: dict[str, Any]) -> list[dict[str, Any]]:
    return list(split["items"])


CHECKPOINT_SCHEMA_VERSION = {"CascHyper-RAG": 3, "Hyper-RAG": 2}
MATCHED_EVIDENCE_PROTOCOL = "rq6_matched_source_text_budget_v1"
FINAL_CONTEXT_PROTOCOL = "rq6_generator_visible_context_v1"


def _checkpoint_schema_version(method: str) -> int:
    try:
        return CHECKPOINT_SCHEMA_VERSION[method]
    except KeyError as error:
        raise ValueError(f"Unsupported RQ6 trace method: {method}") from error


def _checkpoint_path(checkpoint_root: Path, method: str, question_id: str) -> Path:
    return checkpoint_root / method / f"{question_id}.json"


def _checkpoint_trace(
    checkpoint_root: Path,
    method: str,
    item: dict[str, Any],
    trace: dict[str, Any],
    sentence_ids: set[str],
    manifest_digest: str,
    split_digest: str,
    protocol: str = "native",
) -> None:
    """Atomically persist one completed question so an interrupted run can resume."""
    errors = validate_traces([trace], sentence_ids, {item["question_id"]}, method)
    if errors:
        raise ValueError("Refusing to checkpoint invalid trace:\n- " + "\n- ".join(errors))
    destination = _checkpoint_path(checkpoint_root, method, item["question_id"])
    payload = {
        "schema_version": _checkpoint_schema_version(method),
        "method": method,
        "question_id": item["question_id"],
        "stage": item["stage"],
        "manifest_sha256": manifest_digest,
        "split_sha256": split_digest,
        "protocol": protocol,
        "trace": trace,
    }
    temporary = destination.with_suffix(".json.tmp")
    if temporary.exists():
        temporary.unlink()
    try:
        write_json(temporary, payload)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_checkpoint_trace(
    checkpoint_root: Path,
    method: str,
    item: dict[str, Any],
    sentence_ids: set[str],
    manifest_digest: str,
    split_digest: str,
    protocol: str = "native",
) -> dict[str, Any] | None:
    """Return a compatible, validated trace, otherwise force that question to rerun."""
    path = _checkpoint_path(checkpoint_root, method, item["question_id"])
    if not path.exists():
        return None
    try:
        checkpoint = read_json(path)
        if {
            "schema_version": checkpoint.get("schema_version"),
            "method": checkpoint.get("method"),
            "question_id": checkpoint.get("question_id"),
            "stage": checkpoint.get("stage"),
            "manifest_sha256": checkpoint.get("manifest_sha256"),
            "split_sha256": checkpoint.get("split_sha256"),
            "protocol": checkpoint.get("protocol", "native"),
        } != {
            "schema_version": _checkpoint_schema_version(method),
            "method": method,
            "question_id": item["question_id"],
            "stage": item["stage"],
            "manifest_sha256": manifest_digest,
            "split_sha256": split_digest,
            "protocol": protocol,
        }:
            return None
        trace = checkpoint.get("trace")
        if not isinstance(trace, dict):
            return None
        if validate_traces([trace], sentence_ids, {item["question_id"]}, method):
            return None
        return trace
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _load_v81_engine(hyperrag_root: Path, contexts: list[str], runtime_dir: Path):
    _remove_hyperrag_modules()
    sys.path.insert(0, str(hyperrag_root))
    sys.path.insert(0, str(hyperrag_root / "Hyper-RAG"))
    # The persisted v8.1 pickle records this historical module name.
    spec = importlib.util.spec_from_file_location("rag_v81", hyperrag_root / "rag(8.1).py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules["rag_v81"] = module
    spec.loader.exec_module(module)
    module.CACHE_DIR = str(hyperrag_root / "hyperedge_cache")
    module.PROMPT_CACHE_DIR = str(runtime_dir / "v81_prompt_cache")
    engine = module.HyperRAG_Attention_v81()
    raw_text = "\n\n".join(contexts)
    cache_path = module.get_cache_path(raw_text, prefix="hyperrag_v81", model="deepseek-v4-flash", max_token_size=1200, overlap_token_size=100)
    if not module.load_engine_state(engine, cache_path, max_token_size=1200, overlap_token_size=100):
        raise RuntimeError(f"Missing compatible CascHyper-RAG index cache: {cache_path}")
    return engine, cache_path


def _engine_chunk_provenance(engine: Any, chunk_id: int, mapper: CanonicalSentenceMapper, cache: dict[int, tuple[int, int, str]]) -> tuple[int, int, str]:
    if chunk_id not in cache:
        chunk_text = engine.chunks[chunk_id].text
        start, end = mapper.engine_chunk_span(chunk_text)
        cache[chunk_id] = (start, end, chunk_text)
    return cache[chunk_id]


def _precompute_engine_chunk_provenance(engine: Any, mapper: CanonicalSentenceMapper) -> dict[int, tuple[int, int, str]]:
    """Locate ordered v8.1 token chunks in one forward scan of the frozen corpus."""
    cache: dict[int, tuple[int, int, str]] = {}
    cursor = 0
    for chunk_id in sorted(engine.chunks):
        text = engine.chunks[chunk_id].text
        normalized = _normalized(text)
        anchor = normalized[: min(120, len(normalized))]
        position = mapper.normalized_combined.find(anchor, cursor)
        while position >= 0 and mapper.normalized_combined[position : position + len(normalized)] != normalized:
            position = mapper.normalized_combined.find(anchor, position + 1)
        if position < 0:
            raise ProvenanceError(f"CascHyper-RAG chunk {chunk_id} is absent from the frozen corpus")
        start, end = mapper.engine_chunk_span_from_normalized_position(position, len(normalized))
        cache[chunk_id] = (start, end, text)
        cursor = position
    return cache


def _selected_v81_topics(captured_stdout: str) -> list[int]:
    match = re.search(r"Selected Topics:\s*\[([^\]]*)\]", captured_stdout)
    if match is None:
        raise RuntimeError("CascHyper-RAG did not expose its topic-routing decision")
    values = match.group(1).strip()
    return [] if not values else [int(value.strip()) for value in values.split(",")]


def _latent_topic_id(topic_id: int) -> str:
    return f"latent:{int(topic_id)}"


def _selected_topic_candidate_chunk_ids(engine: Any, selected_topic_ids: set[int]) -> set[int]:
    """Return the fixed Casc chunks admitted by at least one selected latent topic."""
    return {
        int(chunk_id)
        for chunk_id, chunk in engine.chunks.items()
        if selected_topic_ids.intersection(int(item) for item in chunk.topic_memberships)
    }


def _latent_topic_manifest(
    engine: Any, mapper: CanonicalSentenceMapper, cache: dict[int, tuple[int, int, str]]
) -> dict[str, Any]:
    """Create the immutable latent-topic candidate-space contract for Topic Coverage."""
    topic_chunks: dict[int, set[int]] = {}
    for chunk_id, chunk in engine.chunks.items():
        for topic_id in chunk.topic_memberships:
            topic_chunks.setdefault(int(topic_id), set()).add(int(chunk_id))
    topics: dict[str, dict[str, Any]] = {}
    for topic_id, chunk_ids in sorted(topic_chunks.items()):
        source_sentence_ids: set[str] = set()
        for chunk_id in chunk_ids:
            start, end, _ = _engine_chunk_provenance(engine, chunk_id, mapper, cache)
            source_sentence_ids.update(mapper.source_sentence_ids_for_span(start, end))
        if not source_sentence_ids:
            raise ProvenanceError(f"Latent CascHyper-RAG topic {topic_id} has no canonical source spans")
        topics[_latent_topic_id(topic_id)] = {
            "chunk_count": len(chunk_ids),
            "chunk_ids": sorted(chunk_ids),
            "source_sentence_ids": sorted(source_sentence_ids),
        }
    if not topics:
        raise ProvenanceError("CascHyper-RAG index contains no latent topic memberships")
    return {
        "schema_version": 1,
        "method": "CascHyper-RAG",
        "topic_namespace": "v81_latent_topic",
        "total_chunk_count": len(engine.chunks),
        "topics": topics,
    }


def _unit_bridges(rank: int, entities: list[str], source_sentence_ids: list[str]) -> list[dict[str, Any]]:
    """Export each entity connection evidenced inside one retrieved source unit.

    A v8.1 sentence can cover multiple canonical original-document sentences.  For
    RQ6's undirected bridge definition, any extracted entity in that retrieved
    unit connects each distinct pair of its covered canonical sentence spans.
    """
    sentence_ids = list(dict.fromkeys(str(sentence_id) for sentence_id in source_sentence_ids))
    entity_names = list(dict.fromkeys(str(entity).strip() for entity in entities if str(entity).strip()))
    return [
        {
            "rank": rank,
            "canonical_entity": entity,
            "from_sentence_id": sentence_ids[left_index],
            "to_sentence_id": sentence_ids[right_index],
        }
        for entity in entity_names
        for left_index in range(len(sentence_ids))
        for right_index in range(left_index + 1, len(sentence_ids))
    ]


def _truncate_casc_chunks(chunks: list[dict[str, Any]], token_budget: int, chunk_text: Any, token_count: Any) -> tuple[list[dict[str, Any]], int]:
    """Keep whole Casc chunks in native rank order under the shared source-text cap."""
    selected: list[dict[str, Any]] = []
    used = 0
    for item in chunks:
        tokens = int(token_count(str(chunk_text(item))))
        if used + tokens > token_budget:
            break
        selected.append(item)
        used += tokens
    return selected, used


def _final_context_units(*groups: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Record exactly the source spans carried by assembled generation evidence."""
    return [
        {"rank": rank, "source_sentence_ids": list(unit["source_sentence_ids"])}
        for rank, unit in enumerate((unit for group in groups for unit in group), start=1)
    ]


async def _capture_casc_generation_prompt(engine: Any, question: str, results: dict[str, Any]) -> str:
    """Execute the native v8.1 prompt assembler without issuing a generation call."""
    module = sys.modules[engine.__class__.__module__]
    original = module.get_completion_api
    prompts: list[str] = []

    async def capture(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    module.get_completion_api = capture
    try:
        await engine.generate_answer(question, results)
    finally:
        module.get_completion_api = original
    if len(prompts) != 1:
        raise ProvenanceError("CascHyper-RAG did not assemble exactly one generation prompt")
    return prompts[0]


def _source_rows_from_hyper_context(contexts: list[str], mapper: CanonicalSentenceMapper) -> list[dict[str, Any]]:
    """Map source rows from Hyper-RAG's complete native generation context."""
    units: list[dict[str, Any]] = []
    seen: set[str] = set()
    for context in contexts:
        match = re.search(r"-----Sources-----\s*```csv\s*(.*?)\s*```", context, flags=re.DOTALL)
        if match is None:
            continue
        for row in csv.DictReader(io.StringIO(match.group(1))):
            text = str(next(
                (value for key, value in row.items() if key is not None and key.strip().casefold() == "content"),
                "",
            )).strip()
            if len(text) >= 2 and text.startswith('"') and text.endswith('"'):
                text = next(csv.reader([text]))[0]
            if not text or text in seen:
                continue
            seen.add(text)
            document_id, _ = mapper.document_for_text(text)
            units.append({"source_sentence_ids": mapper.chunk_sentence_ids(document_id, text)})
    return _final_context_units(units)


async def _capture_hyper_final_context_units(rag: Any, operate: Any, question: str, query_param: Any, mapper: CanonicalSentenceMapper) -> list[dict[str, Any]]:
    """Capture the complete native context passed from Hyper-RAG retrieval to generation."""
    captured: list[str] = []
    original = operate.combine_contexts

    def capture(relation_context: str, entity_context: str) -> str:
        combined = original(relation_context, entity_context)
        captured.append(combined)
        return combined

    operate.combine_contexts = capture
    try:
        await rag.aquery(question, query_param)
    finally:
        operate.combine_contexts = original
    return _source_rows_from_hyper_context(captured, mapper)


async def _capture_native_hyper_evidence(
    rag: Any,
    operate: Any,
    question: str,
    query_param: Any,
    mapper: CanonicalSentenceMapper,
    chunk_occurrences: dict[str, tuple[int, int]],
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """Capture native post-truncation source units and VDB candidates in one query."""
    counts = {"relation_vdb_candidate_count": 0, "entity_vdb_candidate_count": 0}
    original_relation_query = rag.relationships_vdb.query
    original_entity_query = rag.entities_vdb.query
    original_text_get = rag.text_chunks.get_by_id
    original_entity_sources = operate._find_most_related_text_unit_from_entities
    original_relation_sources = operate._find_related_text_unit_from_relationships
    fetched: dict[int, tuple[str, dict[str, Any]]] = {}
    selected: list[tuple[str, dict[str, Any]]] = []

    async def capture_relation_query(*args: Any, **kwargs: Any) -> Any:
        results = await original_relation_query(*args, **kwargs)
        counts["relation_vdb_candidate_count"] = len(results)
        return results

    async def capture_entity_query(*args: Any, **kwargs: Any) -> Any:
        results = await original_entity_query(*args, **kwargs)
        counts["entity_vdb_candidate_count"] = len(results)
        return results

    async def capture_text_get(chunk_id: str, *args: Any, **kwargs: Any) -> Any:
        data = await original_text_get(chunk_id, *args, **kwargs)
        if isinstance(data, dict):
            fetched[id(data)] = (str(chunk_id), data)
        return data

    async def capture_selected_sources(original: Any, *args: Any, **kwargs: Any) -> Any:
        units = await original(*args, **kwargs)
        for data in units:
            source = fetched.get(id(data))
            if source is None:
                raise ProvenanceError("Hyper-RAG selected a source unit without a stable text-unit ID")
            selected.append(source)
        return units

    async def capture_entity_sources(*args: Any, **kwargs: Any) -> Any:
        return await capture_selected_sources(original_entity_sources, *args, **kwargs)

    async def capture_relation_sources(*args: Any, **kwargs: Any) -> Any:
        return await capture_selected_sources(original_relation_sources, *args, **kwargs)

    rag.relationships_vdb.query = capture_relation_query
    rag.entities_vdb.query = capture_entity_query
    rag.text_chunks.get_by_id = capture_text_get
    operate._find_most_related_text_unit_from_entities = capture_entity_sources
    operate._find_related_text_unit_from_relationships = capture_relation_sources
    try:
        await rag.aquery(question, query_param)
    finally:
        rag.relationships_vdb.query = original_relation_query
        rag.entities_vdb.query = original_entity_query
        rag.text_chunks.get_by_id = original_text_get
        operate._find_most_related_text_unit_from_entities = original_entity_sources
        operate._find_related_text_unit_from_relationships = original_relation_sources
    units = []
    seen: set[str] = set()
    for chunk_id, data in selected:
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        units.append({"source_sentence_ids": _baseline_source_sentence_ids(mapper, chunk_id, data, chunk_occurrences)})
    return _final_context_units(units), counts


def _final_context_trace_unit(candidate: dict[str, Any], rank: int) -> dict[str, Any]:
    """Serialize one budgeted source item with enough data to audit the token cap."""
    return {
        "rank": rank,
        "source_sentence_ids": list(candidate["source_sentence_ids"]),
        "source_text": str(candidate["text"]),
        "source_token_count": int(candidate["source_token_count"]),
        "source_text_sha256": hashlib.sha256(str(candidate["text"]).encode("utf-8")).hexdigest(),
        "source_origin": str(candidate["source_origin"]),
    }


def _replace_hyper_source_section(context: str, selected: list[dict[str, Any]]) -> str:
    """Keep Hyper's native entity/relation sections while replacing only Sources."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["id", "content"])
    for index, item in enumerate(selected, start=1):
        writer.writerow([index, item["text"]])
    replacement = buffer.getvalue().strip()
    pattern = r"(-----Sources-----\s*```csv\s*)(.*?)(\s*```)"
    updated, substitutions = re.subn(pattern, lambda match: match.group(1) + replacement + match.group(3), context, count=1, flags=re.DOTALL)
    if substitutions != 1:
        raise ProvenanceError("Hyper-RAG native context has no replaceable Sources section")
    return updated


async def _matched_final_casc_trace(
    engine: Any,
    question_id: str,
    question: str,
    mapper: CanonicalSentenceMapper,
    chunk_cache: dict[int, tuple[int, int, str]],
    token_budget: int,
) -> dict[str, Any]:
    """Capture Casc's actual generation evidence after a shared source-text cap."""
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        results = await engine.search(question, top_k_chunks=5, top_k_sents=10, enable_multi_hop=True)
        results = await engine.verify_results(question, results)

    import tiktoken

    encoder = tiktoken.encoding_for_model("gpt-4o")
    candidates: list[dict[str, Any]] = []
    for item in results["top_chunks"]:
        chunk_id = int(item["chunk_id"])
        start, end, _ = _engine_chunk_provenance(engine, chunk_id, mapper, chunk_cache)
        candidates.append({
            "source_id": f"coarse:{chunk_id}",
            "source_origin": "coarse_chunk",
            "text": engine.chunks[chunk_id].text,
            "source_sentence_ids": mapper.source_sentence_ids_for_span(start, end),
            "item": item,
        })
    for hop_name in ("hop1", "hop2"):
        for item in results[hop_name]:
            raw_chunk_id = item.get("chunk_id", item.get("chunk_ids", [None])[0])
            if raw_chunk_id is None:
                raise ProvenanceError("CascHyper-RAG sentence is missing its parent chunk ID")
            chunk_id = int(raw_chunk_id)
            parent_start, _, parent_text = _engine_chunk_provenance(engine, chunk_id, mapper, chunk_cache)
            # The RQ6 gold is anchored to original-document spans.  Keep the
            # retrieved raw sentence at this controlled source-evidence
            # boundary instead of a window-expanded rendering that has no
            # one-to-one canonical span.
            text = str(item["text"])
            candidates.append({
                "source_id": f"{hop_name}:{item.get('sent_id')}",
                "source_origin": hop_name,
                "text": text,
                "source_sentence_ids": mapper.engine_sentence_ids(parent_start, parent_text, text),
                "item": item,
            })
    selected, used_tokens = select_complete_source_units(candidates, token_budget, lambda text: len(encoder.encode(text)))
    selected_top_chunks = [item["item"] for item in selected if item["source_origin"] == "coarse_chunk"]
    selected_hop1 = []
    selected_hop2 = []
    for candidate in selected:
        if candidate["source_origin"] not in {"hop1", "hop2"}:
            continue
        raw_item = candidate["item"]
        prompt_item = {key: value for key, value in raw_item.items() if key != "expanded_text"}
        (selected_hop1 if candidate["source_origin"] == "hop1" else selected_hop2).append(prompt_item)
    limited_results = {
        **results,
        "top_chunks": selected_top_chunks,
        "hop1": selected_hop1,
        "hop2": selected_hop2,
    }
    generation_prompt = await _capture_casc_generation_prompt(engine, question, limited_results)
    if any(item["text"] not in generation_prompt for item in selected):
        raise ProvenanceError("CascHyper-RAG generation prompt omitted a budgeted source evidence item")
    chunks = [
        {"rank": rank, "source_sentence_ids": list(item["source_sentence_ids"])}
        for rank, item in enumerate((item for item in selected if item["source_origin"] == "coarse_chunk"), start=1)
    ]
    evidence = [
        {"rank": rank, "source_sentence_ids": list(item["source_sentence_ids"])}
        for rank, item in enumerate((item for item in selected if item["source_origin"] != "coarse_chunk"), start=1)
    ]
    return {
        "question_id": question_id,
        "method": "CascHyper-RAG",
        "retrieved_chunks": chunks,
        "retrieved_evidence_units": evidence,
        "retrieved_bridge_entities": [],
        "retrieved_bridges": [],
        "final_context_units": [_final_context_trace_unit(item, rank) for rank, item in enumerate(selected, start=1)],
        "trace_diagnostics": {
            "final_context_protocol": MATCHED_FINAL_CONTEXT_PROTOCOL,
            "source_text_budget_tokens": token_budget,
            "selected_source_tokens": used_tokens,
            "candidate_source_unit_count": len(candidates),
            "final_context_unit_count": len(selected),
            "source_order": "coarse_top5_then_hop1_then_hop2",
            "generation_prompt_sha256": hashlib.sha256(generation_prompt.encode("utf-8")).hexdigest(),
        },
    }


async def _matched_final_hyper_trace(
    rag: Any,
    operate: Any,
    question_id: str,
    question: str,
    query_param: Any,
    mapper: CanonicalSentenceMapper,
    chunk_occurrences: dict[str, tuple[int, int]],
    token_budget: int,
) -> dict[str, Any]:
    """Capture Hyper's native track sources and globally cap them before generation."""
    original_text_get = rag.text_chunks.get_by_id
    original_entity_sources = operate._find_most_related_text_unit_from_entities
    original_relation_sources = operate._find_related_text_unit_from_relationships
    original_combine = operate.combine_contexts
    original_truncate = operate.truncate_list_by_token_size
    fetched: dict[int, tuple[str, dict[str, Any]]] = {}
    selected_tracks: dict[str, list[tuple[str, dict[str, Any]]]] = {"relation": [], "entity": []}
    captured_contexts: list[str] = []
    final_candidates: list[dict[str, Any]] = []
    used_tokens = 0

    async def capture_text_get(chunk_id: str, *args: Any, **kwargs: Any) -> Any:
        data = await original_text_get(chunk_id, *args, **kwargs)
        if isinstance(data, dict):
            fetched[id(data)] = (str(chunk_id), data)
        return data

    async def capture_track(track: str, original: Any, *args: Any, **kwargs: Any) -> Any:
        units = await original(*args, **kwargs)
        for data in units:
            source = fetched.get(id(data))
            if source is None:
                raise ProvenanceError("Hyper-RAG selected a source unit without a stable text-unit ID")
            selected_tracks[track].append(source)
        return units

    async def capture_entity_sources(*args: Any, **kwargs: Any) -> Any:
        return await capture_track("entity", original_entity_sources, *args, **kwargs)

    async def capture_relation_sources(*args: Any, **kwargs: Any) -> Any:
        return await capture_track("relation", original_relation_sources, *args, **kwargs)

    def retain_pre_assembly_source_candidates(list_data: list[Any], key: Any, max_token_size: int) -> list[Any]:
        """Bypass only Hyper's per-track source cap; all other native caps remain live."""
        is_source_unit_list = bool(list_data) and all(
            isinstance(item, dict)
            and isinstance(item.get("data"), dict)
            and isinstance(item["data"].get("content"), str)
            for item in list_data
        )
        if is_source_unit_list and max_token_size == query_param.max_token_for_text_unit:
            return list_data
        return original_truncate(list_data, key, max_token_size)

    def capture_combine(relation_context: str, entity_context: str) -> str:
        nonlocal final_candidates, used_tokens
        candidates: list[dict[str, Any]] = []
        seen_chunk_ids: set[str] = set()
        seen_texts: set[str] = set()
        for track in ("relation", "entity"):
            for chunk_id, data in selected_tracks[track]:
                text = str(data["content"])
                if chunk_id in seen_chunk_ids or text in seen_texts:
                    continue
                seen_chunk_ids.add(chunk_id)
                seen_texts.add(text)
                candidates.append({
                    "source_id": chunk_id,
                    "source_origin": f"{track}_track",
                    "text": text,
                    "source_sentence_ids": _baseline_source_sentence_ids(mapper, chunk_id, data, chunk_occurrences),
                })
        final_candidates, used_tokens = select_complete_source_units(
            candidates, token_budget, lambda text: len(operate.encode_string_by_tiktoken(text))
        )
        combined = _replace_hyper_source_section(original_combine(relation_context, entity_context), final_candidates)
        captured_contexts.append(combined)
        return combined

    rag.text_chunks.get_by_id = capture_text_get
    operate._find_most_related_text_unit_from_entities = capture_entity_sources
    operate._find_related_text_unit_from_relationships = capture_relation_sources
    operate.combine_contexts = capture_combine
    operate.truncate_list_by_token_size = retain_pre_assembly_source_candidates
    try:
        await rag.aquery(question, query_param)
    finally:
        rag.text_chunks.get_by_id = original_text_get
        operate._find_most_related_text_unit_from_entities = original_entity_sources
        operate._find_related_text_unit_from_relationships = original_relation_sources
        operate.combine_contexts = original_combine
        operate.truncate_list_by_token_size = original_truncate
    if len(captured_contexts) != 1:
        raise ProvenanceError("Hyper-RAG did not assemble exactly one combined generation context")
    context = captured_contexts[0]
    if any(item["text"] not in context for item in final_candidates):
        raise ProvenanceError("Hyper-RAG generation context omitted a budgeted source evidence item")
    units = [_final_context_trace_unit(item, rank) for rank, item in enumerate(final_candidates, start=1)]
    return {
        "question_id": question_id,
        "method": "Hyper-RAG",
        "retrieved_chunks": [{"rank": unit["rank"], "source_sentence_ids": unit["source_sentence_ids"]} for unit in units],
        "retrieved_evidence_units": [{"rank": unit["rank"], "source_sentence_ids": unit["source_sentence_ids"]} for unit in units],
        "retrieved_bridge_entities": [],
        "retrieved_bridges": [],
        "final_context_units": units,
        "trace_diagnostics": {
            "final_context_protocol": MATCHED_FINAL_CONTEXT_PROTOCOL,
            "source_text_budget_tokens": token_budget,
            "selected_source_tokens": used_tokens,
            "candidate_source_unit_count": sum(len(items) for items in selected_tracks.values()),
            "final_context_unit_count": len(units),
            "source_order": "relation_track_then_entity_track",
            "native_per_track_source_cap_tokens": query_param.max_token_for_text_unit,
            "pre_assembly_source_cap_adapter": "bypass_native_per_track_cap_then_apply_shared_final_cap",
            "generation_context_sha256": hashlib.sha256(context.encode("utf-8")).hexdigest(),
        },
    }


async def _casc_trace_for_query(engine: Any, question_id: str, question: str, mapper: CanonicalSentenceMapper, chunk_cache: dict[int, tuple[int, int, str]], source_token_budget: int | None = None, capture_final_context: bool = False, native_evidence: bool = False) -> dict[str, Any]:
    if native_evidence and bool(getattr(engine, "enable_consistency_verifier", False)) != native_retrieval_config("CascHyper-RAG")["consistency_verification"]:
        raise RuntimeError("CascHyper-RAG verifier state does not match the frozen RQ1/RQ2 configuration")
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured):
        results = await engine.search(question, top_k_chunks=5, top_k_sents=10, enable_multi_hop=True)
        if capture_final_context or native_evidence:
            results = await engine.verify_results(question, results)
        if capture_final_context:
            generation_prompt = await _capture_casc_generation_prompt(engine, question, results)
    selected_tids = _selected_v81_topics(captured.getvalue())
    selected_topic_ids = {_latent_topic_id(item) for item in selected_tids}
    candidate_chunks = _selected_topic_candidate_chunk_ids(engine, set(selected_tids))
    selected_top_chunks = results["top_chunks"]
    selected_source_tokens = None
    if source_token_budget is not None:
        import tiktoken

        encoder = tiktoken.encoding_for_model("gpt-4o")
        selected_top_chunks, selected_source_tokens = _truncate_casc_chunks(
            results["top_chunks"],
            source_token_budget,
            lambda item: engine.chunks[int(item["chunk_id"])].text,
            lambda text: len(encoder.encode(text)),
        )
    selected_chunk_ids = {int(item["chunk_id"]) for item in selected_top_chunks}
    chunks = []
    for rank, item in enumerate(selected_top_chunks, start=1):
        start, end, _ = _engine_chunk_provenance(engine, int(item["chunk_id"]), mapper, chunk_cache)
        chunk_record = {"rank": rank, "score": float(item["score"]), "source_sentence_ids": mapper.source_sentence_ids_for_span(start, end)}
        if capture_final_context:
            chunk_record["_final_text"] = engine.chunks[int(item["chunk_id"])].text
        chunks.append(chunk_record)
    evidence = []
    unmapped_evidence = []
    hop1_sentence_ids: dict[str, list[str]] = {}
    for item in results["hop1"] + results["hop2"]:
        raw_chunk_id = item.get("chunk_id")
        if raw_chunk_id is None:
            raw_chunk_id = item.get("chunk_ids", [None])[0]
        if raw_chunk_id is None:
            raise ProvenanceError("CascHyper-RAG sentence is missing its parent chunk ID")
        chunk_id = int(raw_chunk_id)
        if not capture_final_context and not native_evidence and chunk_id not in selected_chunk_ids:
            continue
        parent_start, _, parent_text = _engine_chunk_provenance(engine, chunk_id, mapper, chunk_cache)
        generation_text = item.get("expanded_text", item["text"]) if capture_final_context or native_evidence else item["text"]
        try:
            source_ids = mapper.engine_sentence_ids(parent_start, parent_text, generation_text)
        except ProvenanceError as error:
            unmapped_evidence.append({"hop": item.get("hop", 1), "score": float(item["score"]), "text": generation_text, "reason": str(error)})
            continue
        evidence_record = {
            "score": float(item["score"]),
            "hop": item.get("hop", 1),
            "source_sentence_ids": source_ids,
            "_via": item.get("via"),
            "_entities": [entity.name for entity in item["sentence"].entities],
        }
        if capture_final_context:
            evidence_record["_final_text"] = generation_text
        evidence.append(evidence_record)
        if item.get("hop", 1) == 1:
            hop1_sentence_ids[str(item["sent_id"])] = source_ids
    evidence.sort(key=lambda item: item["score"], reverse=True)
    bridges = []
    for rank, item in enumerate(evidence, start=1):
        item["rank"] = rank
        via = item.pop("_via")
        bridges.extend(_unit_bridges(rank, item.pop("_entities"), item["source_sentence_ids"]))
        if via:
            for candidate in results["hop1"]:
                if any(entity.name == via for entity in candidate["sentence"].entities):
                    for left in hop1_sentence_ids.get(str(candidate["sent_id"]), []):
                        for right in item["source_sentence_ids"]:
                            bridges.append({"rank": rank, "canonical_entity": via, "from_sentence_id": left, "to_sentence_id": right})
    diagnostics = {
        "returned_evidence_units": len(results["hop1"]) + len(results["hop2"]),
        "mapped_evidence_units": len(evidence),
        "unmapped_evidence_units": unmapped_evidence,
        "topic_candidate_chunk_count": len(candidate_chunks),
        "topic_candidate_total_chunk_count": len(engine.chunks),
    }
    if source_token_budget is not None:
        diagnostics.update({
            "evaluation_protocol": MATCHED_EVIDENCE_PROTOCOL,
            "source_text_budget_tokens": source_token_budget,
            "selected_source_tokens": selected_source_tokens,
            "selected_source_unit_count": len(selected_top_chunks),
            "coarse_retrieval_contract": "top-5 cached chunks; each cache chunk was built with max_token_size=1200",
        })
    if native_evidence:
        diagnostics.update({
            "rq6_protocol": NATIVE_EVIDENCE_PROTOCOL,
            "native_retrieval_config": native_retrieval_config("CascHyper-RAG"),
            "selected_chunk_count": len(chunks),
            "selected_hop1_count": sum(item["hop"] == 1 for item in evidence),
            "selected_hop2_count": sum(item["hop"] == 2 for item in evidence),
            "selected_unique_entities": len({
                entity.name for item in results["hop1"] + results["hop2"] for entity in item["sentence"].entities
            }),
            "selected_bridge_count": len(bridges),
        })
    final_units = []
    if capture_final_context:
        for item in chunks + evidence:
            text = item.pop("_final_text")
            if text not in generation_prompt:
                raise ProvenanceError("CascHyper-RAG generation prompt omitted a selected source evidence item")
            final_units.append({"source_sentence_ids": item["source_sentence_ids"]})
    trace = {"question_id": question_id, "method": "CascHyper-RAG", "selected_topic_ids": sorted(selected_topic_ids), "retrieved_chunks": chunks, "retrieved_evidence_units": evidence, "retrieved_bridge_entities": sorted({edge["canonical_entity"] for edge in bridges}), "retrieved_bridges": bridges, "trace_diagnostics": diagnostics}
    if capture_final_context:
        trace["final_context_units"] = _final_context_units(final_units)
        diagnostics["final_context_protocol"] = FINAL_CONTEXT_PROTOCOL
        diagnostics["final_context_unit_count"] = len(trace["final_context_units"])
        diagnostics["generation_prompt_sha256"] = hashlib.sha256(generation_prompt.encode("utf-8")).hexdigest()
    return trace


def _load_hyperrag_main(hyperrag_root: Path, runtime_dir: Path, source_token_budget: int = 1200, relation_context_budget: int | None = None):
    _remove_hyperrag_modules()
    for unwanted in (str(hyperrag_root / "Hyper-RAG"),):
        while unwanted in sys.path:
            sys.path.remove(unwanted)
    sys.path.insert(0, str(hyperrag_root))
    sys.path.insert(0, str(hyperrag_root / "Hyper-RAG-main"))
    from experiment_config import EMB_DIM
    from run_experiment import global_emb_func_list, global_llm_func
    from hyperrag import HyperRAG, QueryParam
    import hyperrag.hyperrag as hyperrag_module
    import hyperrag.operate as operate
    import hyperrag.utils as hyperrag_utils
    from hyperrag.utils import EmbeddingFunc
    working_dir = hyperrag_root / "Hyper-RAG-main" / "caches" / "deepseek-v4-flash" / "physics" / "index"
    if not working_dir.exists():
        raise RuntimeError(f"Missing compatible Hyper-RAG index cache: {working_dir}")
    hyperrag_module.set_logger = lambda _ignored: hyperrag_utils.set_logger(str(runtime_dir / "hyperrag_main.log"))
    rag = HyperRAG(working_dir=str(working_dir), llm_model_func=global_llm_func, embedding_func=EmbeddingFunc(embedding_dim=EMB_DIM, max_token_size=8192, func=global_emb_func_list), chunk_token_size=1200, chunk_overlap_token_size=100, llm_model_max_async=4, entity_extract_max_gleaning=0, enable_llm_cache=False)
    query_kwargs = {"mode": "hyper", "top_k": 10, "max_token_for_text_unit": source_token_budget, "only_need_context": True}
    if relation_context_budget is not None:
        query_kwargs["max_token_for_relation_context"] = relation_context_budget
    return rag, QueryParam(**query_kwargs), operate


def _baseline_chunk_document(mapper: CanonicalSentenceMapper, data: dict[str, Any]) -> str:
    full_doc_id = str(data.get("full_doc_id", ""))
    if not full_doc_id:
        raise ProvenanceError("Hyper-RAG text unit is missing full_doc_id")
    for document_id, document_text in mapper.document_texts.items():
        if hashlib.md5(document_text.encode("utf-8")).hexdigest() in full_doc_id:
            return document_id
    document_id, _ = mapper.document_for_text(str(data["content"]))
    return document_id


def _baseline_chunk_occurrences(index_directory: Path) -> dict[str, tuple[int, int]]:
    """Return each cached Hyper-RAG chunk's stable position among identical source text."""
    chunks = read_json(index_directory / "kv_store_text_chunks.json")
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = {}
    for chunk_id, data in chunks.items():
        grouped.setdefault((str(data["full_doc_id"]), str(data["content"])), []).append(
            (int(data["chunk_order_index"]), str(chunk_id))
        )
    occurrences: dict[str, tuple[int, int]] = {}
    for values in grouped.values():
        values.sort()
        for occurrence_index, (_, chunk_id) in enumerate(values):
            occurrences[chunk_id] = (occurrence_index, len(values))
    return occurrences


def _baseline_source_sentence_ids(
    mapper: CanonicalSentenceMapper,
    chunk_id: str,
    data: dict[str, Any],
    occurrences: dict[str, tuple[int, int]],
) -> list[str]:
    """Map a source unit through its original Hyper-RAG chunk ID, never text guessing."""
    document_id = _baseline_chunk_document(mapper, data)
    text = str(data["content"])
    try:
        return mapper.chunk_sentence_ids(document_id, text)
    except ProvenanceError:
        occurrence = occurrences.get(chunk_id)
        if occurrence is None:
            raise ProvenanceError("Hyper-RAG selected a text unit absent from its frozen text-unit cache")
        return mapper.chunk_sentence_ids_at_occurrence(document_id, text, *occurrence)


def _entity_surface_matches(entity: str, sentence: str) -> bool:
    """Match an entity as a complete phrase, never as a substring of another token."""
    pattern = r"(?<![a-z0-9])" + re.escape(entity.casefold()) + r"(?![a-z0-9])"
    return re.search(pattern, sentence.casefold()) is not None


async def _relation_units(rag: Any, operate: Any, keywords: str, param: Any) -> tuple[list[dict[str, Any]], list[tuple[list[str], list[str]]]]:
    results = await rag.relationships_vdb.query(keywords, top_k=param.top_k)
    raw_edges = await asyncio.gather(*[rag.chunk_entity_relation_hypergraph.get_hyperedge(item["id_set"]) for item in results])
    degrees = await asyncio.gather(*[rag.chunk_entity_relation_hypergraph.hyperedge_degree(item["id_set"]) for item in results])
    edges = [{"id_set": result["id_set"], "rank": degree, **edge} for result, edge, degree in zip(results, raw_edges, degrees) if edge is not None]
    edges.sort(key=lambda item: (item["rank"], item["weight"]), reverse=True)
    edges = operate.truncate_list_by_token_size(edges, key=lambda item: item["description"], max_token_size=param.max_token_for_relation_context)
    candidates: dict[str, dict[str, Any]] = {}
    relations = []
    for order, edge in enumerate(edges):
        ids = operate.split_string_by_multi_markers(edge["source_id"], [operate.GRAPH_FIELD_SEP])
        relations.append(([str(entity) for entity in edge["id_set"]], ids))
        for chunk_id in ids:
            if chunk_id not in candidates:
                data = await rag.text_chunks.get_by_id(chunk_id)
                if data and data.get("content"):
                    candidates[chunk_id] = {"chunk_id": chunk_id, "data": data, "order": order, "track": "relation"}
    selected = sorted(candidates.values(), key=lambda item: item["order"])
    return operate.truncate_list_by_token_size(selected, key=lambda item: item["data"]["content"], max_token_size=param.max_token_for_text_unit), relations


async def _entity_units(rag: Any, operate: Any, keywords: str, param: Any) -> list[dict[str, Any]]:
    results = await rag.entities_vdb.query(keywords, top_k=param.top_k)
    nodes = await asyncio.gather(*[rag.chunk_entity_relation_hypergraph.get_vertex(item["entity_name"]) for item in results])
    node_data = [node for node in nodes if node is not None]
    edge_lists = await asyncio.gather(*[rag.chunk_entity_relation_hypergraph.get_nbr_e_of_vertex(item["entity_name"]) for item in results if item])
    all_nodes = {entity for edges in edge_lists for edge in (edges or []) for entity in edge}
    neighbor_values = await asyncio.gather(*[rag.chunk_entity_relation_hypergraph.get_vertex(entity) for entity in all_nodes])
    neighbor_sources = {entity: set(operate.split_string_by_multi_markers(data["source_id"], [operate.GRAPH_FIELD_SEP])) for entity, data in zip(all_nodes, neighbor_values) if data and data.get("source_id")}
    candidates: dict[str, dict[str, Any]] = {}
    for order, (node, edges) in enumerate(zip(node_data, edge_lists)):
        for chunk_id in operate.split_string_by_multi_markers(node["source_id"], [operate.GRAPH_FIELD_SEP]):
            if chunk_id in candidates:
                continue
            links = sum(chunk_id in neighbor_sources.get(entity, set()) for edge in (edges or []) for entity in edge)
            data = await rag.text_chunks.get_by_id(chunk_id)
            if data and data.get("content"):
                candidates[chunk_id] = {"chunk_id": chunk_id, "data": data, "order": order, "links": links, "track": "entity"}
    selected = sorted(candidates.values(), key=lambda item: (item["order"], -item["links"]))
    return operate.truncate_list_by_token_size(selected, key=lambda item: item["data"]["content"], max_token_size=param.max_token_for_text_unit)


def _truncate_hyper_units(units: list[tuple[str, dict[str, Any]]], token_budget: int, token_count: Any) -> tuple[list[tuple[str, dict[str, Any]]], int]:
    """Keep the native relation-then-entity order under one shared source-text budget."""
    selected: list[tuple[str, dict[str, Any]]] = []
    used = 0
    for unit in units:
        tokens = int(token_count(str(unit[1]["content"])))
        if used + tokens > token_budget:
            break
        selected.append(unit)
        used += tokens
    return selected, used


async def _hyper_bridges(rag: Any, operate: Any, relations: list[tuple[list[str], list[str]]], selected_chunk_ranks: dict[str, int], mapper: CanonicalSentenceMapper) -> list[dict[str, Any]]:
    bridges: list[dict[str, Any]] = []
    for entities, source_chunks in relations:
        selected_ids = [item for item in source_chunks if item in selected_chunk_ranks]
        source_ids: list[str] = []
        for chunk_id in selected_ids:
            data = await rag.text_chunks.get_by_id(chunk_id)
            if data is None:
                raise ProvenanceError("Hyper-RAG relation references a missing text unit")
            document_id = _baseline_chunk_document(mapper, data)
            source_ids.extend(mapper.chunk_sentence_ids(document_id, str(data["content"])))
        rank = max((selected_chunk_ranks[item] for item in selected_ids), default=0)
        for entity in entities:
            endpoint_ids = [sentence_id for sentence_id in dict.fromkeys(source_ids) if _entity_surface_matches(entity, str(mapper.sentence_by_id[sentence_id]["text"]))]
            for left in endpoint_ids:
                for right in endpoint_ids:
                    if left != right:
                        bridges.append({"rank": rank, "canonical_entity": entity, "from_sentence_id": left, "to_sentence_id": right})
    return bridges


async def _native_hyper_trace(rag: Any, operate: Any, query_param: Any, question_id: str, question: str, mapper: CanonicalSentenceMapper, source_token_budget: int | None = None, capture_final_context: bool = False, native_evidence: bool = False, chunk_occurrences: dict[str, tuple[int, int]] | None = None) -> dict[str, Any]:
    if native_evidence:
        config = native_retrieval_config("Hyper-RAG")
        actual = {
            "mode": query_param.mode,
            "top_k_per_track": query_param.top_k,
            "max_token_for_text_unit": query_param.max_token_for_text_unit,
            "max_token_for_relation_context": query_param.max_token_for_relation_context,
        }
        if actual != config:
            raise RuntimeError("Hyper-RAG query parameters do not match the frozen RQ1/RQ2 configuration")
        if chunk_occurrences is None:
            raise RuntimeError("Native Hyper-RAG evidence replay requires the frozen text-unit occurrence map")
        units, candidate_counts = await _capture_native_hyper_evidence(
            rag, operate, question, query_param, mapper, chunk_occurrences,
        )
        diagnostics = {
            "rq6_protocol": NATIVE_EVIDENCE_PROTOCOL,
            "native_retrieval_config": config,
            **candidate_counts,
            "selected_source_unit_count": len(units),
            "selected_bridge_count": 0,
        }
        return {
            "question_id": question_id,
            "method": "Hyper-RAG",
            "retrieved_chunks": units,
            "retrieved_evidence_units": units,
            "retrieved_bridge_entities": [],
            "retrieved_bridges": [],
            "trace_diagnostics": diagnostics,
        }
    if capture_final_context:
        final_units = await _capture_hyper_final_context_units(rag, operate, question, query_param, mapper)
        diagnostics = {
            "final_context_protocol": FINAL_CONTEXT_PROTOCOL,
            "final_context_unit_count": len(final_units),
        }
        return {
            "question_id": question_id,
            "method": "Hyper-RAG",
            "retrieved_chunks": final_units,
            "retrieved_evidence_units": final_units,
            "retrieved_bridge_entities": [],
            "retrieved_bridges": [],
            "final_context_units": final_units,
            "trace_diagnostics": diagnostics,
        }
    prompt = operate.PROMPTS["keywords_extraction"].format(query=question)
    keywords = json.loads(await rag.llm_model_func(prompt))
    relation_keywords = ", ".join(keywords.get("high_level_keywords", []))
    entity_keywords = ", ".join(keywords.get("low_level_keywords", []))
    relation_units, relations = await _relation_units(rag, operate, relation_keywords, query_param) if relation_keywords else ([], [])
    entity_units = await _entity_units(rag, operate, entity_keywords, query_param) if entity_keywords else []
    units = []
    seen = set()
    for item in relation_units + entity_units:
        if item["chunk_id"] not in seen:
            units.append((item["chunk_id"], item["data"]))
            seen.add(item["chunk_id"])
    selected_source_tokens = None
    if source_token_budget is not None:
        units, selected_source_tokens = _truncate_hyper_units(
            units, source_token_budget, lambda text: len(operate.encode_string_by_tiktoken(text))
        )
    if not units:
        # A genuine empty retrieval is a valid system outcome, not an export
        # failure.  Keep the frozen question in the trace so every recall metric
        # correctly assigns zero support to this query.
        diagnostics = {}
        if source_token_budget is not None:
            diagnostics = {
                "evaluation_protocol": MATCHED_EVIDENCE_PROTOCOL,
                "source_text_budget_tokens": source_token_budget,
                "selected_source_tokens": 0,
                "selected_source_unit_count": 0,
                "ranking_protocol": "rq6_track_merge_v1",
            }
        trace = {
            "question_id": question_id,
            "method": "Hyper-RAG",
            "retrieval_status": "empty",
            "retrieved_chunks": [],
            "retrieved_evidence_units": [],
            "retrieved_bridge_entities": [],
            "retrieved_bridges": [],
            "trace_diagnostics": diagnostics,
        }
        if capture_final_context:
            trace["final_context_units"] = await _capture_hyper_final_context_units(rag, operate, question, query_param, mapper)
            diagnostics["final_context_protocol"] = FINAL_CONTEXT_PROTOCOL
            diagnostics["final_context_unit_count"] = len(trace["final_context_units"])
        return trace
    chunks = []
    evidence = []
    selected_chunk_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(units, start=1)}
    for rank, (_, data) in enumerate(units, start=1):
        document_id = _baseline_chunk_document(mapper, data)
        source_ids = mapper.chunk_sentence_ids(document_id, str(data["content"]))
        chunks.append({"rank": rank, "source_sentence_ids": source_ids})
        evidence.append({"rank": rank, "source_sentence_ids": source_ids})
    bridges = await _hyper_bridges(rag, operate, relations, selected_chunk_ranks, mapper)
    diagnostics = {}
    if source_token_budget is not None:
        diagnostics = {
            "evaluation_protocol": MATCHED_EVIDENCE_PROTOCOL,
            "source_text_budget_tokens": source_token_budget,
            "selected_source_tokens": selected_source_tokens,
            "selected_source_unit_count": len(units),
            "ranking_protocol": "rq6_track_merge_v1",
        }
    trace = {"question_id": question_id, "method": "Hyper-RAG", "retrieved_chunks": chunks, "retrieved_evidence_units": evidence, "retrieved_bridge_entities": sorted({edge["canonical_entity"] for edge in bridges}), "retrieved_bridges": bridges, "trace_diagnostics": diagnostics}
    if capture_final_context:
        trace["final_context_units"] = await _capture_hyper_final_context_units(rag, operate, question, query_param, mapper)
        diagnostics["final_context_protocol"] = FINAL_CONTEXT_PROTOCOL
        diagnostics["final_context_unit_count"] = len(trace["final_context_units"])
    return trace


async def _hyper_trace_for_query(rag: Any, operate: Any, query_param: Any, question_id: str, question: str, mapper: CanonicalSentenceMapper, source_token_budget: int | None = None, capture_final_context: bool = False, native_evidence: bool = False, chunk_occurrences: dict[str, tuple[int, int]] | None = None) -> dict[str, Any]:
    return await _native_hyper_trace(rag, operate, query_param, question_id, question, mapper, source_token_budget, capture_final_context, native_evidence, chunk_occurrences)


def _publish_traces(casc_output: str | Path, hyper_output: str | Path, casc_traces: list[dict[str, Any]], hyper_traces: list[dict[str, Any]], sentence_ids: set[str], question_ids: set[str]) -> None:
    for output in (Path(casc_output), Path(hyper_output)):
        if output.exists():
            raise ValueError(f"Trace output already exists and will not be overwritten: {output}")
    for traces, method in ((casc_traces, "CascHyper-RAG"), (hyper_traces, "Hyper-RAG")):
        errors = validate_traces(traces, sentence_ids, question_ids, method)
        if errors:
            raise ValueError("Trace validation failed:\n- " + "\n- ".join(errors))
    temporary = [(Path(casc_output).with_suffix(".json.tmp"), casc_traces), (Path(hyper_output).with_suffix(".json.tmp"), hyper_traces)]
    try:
        for path, traces in temporary:
            if path.exists():
                raise ValueError(f"Temporary trace path already exists: {path}")
            write_json(path, traces)
        for path, _ in temporary:
            os.replace(path, Path(casc_output) if path == temporary[0][0] else Path(hyper_output))
    finally:
        for path, _ in temporary:
            if path.exists():
                path.unlink()


def _publish_trace(output: str | Path, traces: list[dict[str, Any]], sentence_ids: set[str], question_ids: set[str], method: str) -> None:
    """Validate and atomically publish one immutable method trace."""
    destination = Path(output)
    if destination.exists():
        raise ValueError(f"Trace output already exists and will not be overwritten: {destination}")
    errors = validate_traces(traces, sentence_ids, question_ids, method)
    if errors:
        raise ValueError("Trace validation failed:\n- " + "\n- ".join(errors))
    temporary = destination.with_suffix(".json.tmp")
    if temporary.exists():
        raise ValueError(f"Temporary trace path already exists: {temporary}")
    try:
        write_json(temporary, traces)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


async def _replay_casc_trace(manifest_path: str | Path, split_path: str | Path, contexts_path: str | Path, hyperrag_root: str | Path, casc_output: str | Path, checkpoint_directory: str | Path, source_token_budget: int | None = None) -> dict[str, int]:
    """Re-export only CascHyper-RAG from its existing frozen Physics index."""
    manifest, split, contexts = read_json(manifest_path), read_json(split_path), read_json(contexts_path)
    mapper = CanonicalSentenceMapper(manifest, contexts)
    questions, root = _question_items(split), Path(hyperrag_root)
    sentence_ids = {item["sentence_id"] for item in manifest["sentences"]}
    manifest_digest, split_digest = sha256_file(manifest_path), sha256_file(split_path)
    checkpoint_protocol = (
        f"{MATCHED_EVIDENCE_PROTOCOL}:{source_token_budget}"
        if source_token_budget is not None
        else "native"
    )
    runtime_dir = Path(__file__).resolve().parents[1] / ".rq6_runtime"
    runtime_dir.mkdir(exist_ok=True)
    engine, casc_cache = _load_v81_engine(root, contexts, runtime_dir)
    cache_before = _tree_digest(casc_cache)
    chunk_cache = _precompute_engine_chunk_provenance(engine, mapper)
    traces: list[dict[str, Any]] = []
    reused = 0
    checkpoint_root = Path(checkpoint_directory)
    for item in questions:
        checkpoint = _load_checkpoint_trace(checkpoint_root, "CascHyper-RAG", item, sentence_ids, manifest_digest, split_digest, checkpoint_protocol)
        if checkpoint is not None:
            traces.append(checkpoint)
            reused += 1
            continue
        try:
            trace = await _casc_trace_for_query(engine, item["question_id"], item["question"], mapper, chunk_cache, source_token_budget)
            _checkpoint_trace(checkpoint_root, "CascHyper-RAG", item, trace, sentence_ids, manifest_digest, split_digest, checkpoint_protocol)
            traces.append(trace)
        except ProvenanceError as error:
            raise ProvenanceError(f"{item['question_id']}: {error}") from error
    if cache_before != _tree_digest(casc_cache):
        raise RuntimeError("The persisted CascHyper-RAG index changed during trace replay; output was not published")
    _publish_trace(casc_output, traces, sentence_ids, {item["question_id"] for item in questions}, "CascHyper-RAG")
    return {"questions": len(questions), "casc_traces": len(traces), "casc_reused_checkpoints": reused}


async def _replay_traces(manifest_path: str | Path, split_path: str | Path, contexts_path: str | Path, hyperrag_root: str | Path, casc_output: str | Path, hyper_output: str | Path, checkpoint_directory: str | Path, matched_source_token_budget: int | None = None, capture_final_context: bool = False, native_evidence: bool = False) -> dict[str, int]:
    """Query both existing Physics indexes and write validated immutable traces."""
    manifest, split, contexts = read_json(manifest_path), read_json(split_path), read_json(contexts_path)
    mapper = CanonicalSentenceMapper(manifest, contexts)
    questions, root = _question_items(split), Path(hyperrag_root)
    checkpoint_root = Path(checkpoint_directory)
    sentence_ids = {item["sentence_id"] for item in manifest["sentences"]}
    manifest_digest, split_digest = sha256_file(manifest_path), sha256_file(split_path)
    checkpoint_protocol = NATIVE_EVIDENCE_PROTOCOL if native_evidence else (FINAL_CONTEXT_PROTOCOL if capture_final_context else (
        f"{MATCHED_EVIDENCE_PROTOCOL}:{matched_source_token_budget}"
        if matched_source_token_budget is not None
        else "native"
    ))
    runtime_dir = Path(__file__).resolve().parents[1] / ".rq6_runtime"
    runtime_dir.mkdir(exist_ok=True)
    casc_engine, casc_cache = _load_v81_engine(root, contexts, runtime_dir)
    baseline_cache = root / "Hyper-RAG-main" / "caches" / "deepseek-v4-flash" / "physics" / "index"
    cache_before = {"casc": _tree_digest(casc_cache), "hyper": _tree_digest(baseline_cache)}
    chunk_cache = _precompute_engine_chunk_provenance(casc_engine, mapper)
    casc_traces = []
    casc_reused = 0
    for item in questions:
        checkpoint = _load_checkpoint_trace(checkpoint_root, "CascHyper-RAG", item, sentence_ids, manifest_digest, split_digest, checkpoint_protocol)
        if checkpoint is not None:
            casc_traces.append(checkpoint)
            casc_reused += 1
            continue
        try:
            trace = await _casc_trace_for_query(casc_engine, item["question_id"], item["question"], mapper, chunk_cache, matched_source_token_budget, capture_final_context, native_evidence)
            _checkpoint_trace(checkpoint_root, "CascHyper-RAG", item, trace, sentence_ids, manifest_digest, split_digest, checkpoint_protocol)
            casc_traces.append(trace)
        except ProvenanceError as error:
            raise ProvenanceError(f"{item['question_id']}: {error}") from error
    hyper_budget = matched_source_token_budget or 1200
    rag, query_param, operate = _load_hyperrag_main(
        root,
        runtime_dir,
        source_token_budget=hyper_budget,
        relation_context_budget=matched_source_token_budget,
    )
    chunk_occurrences = _baseline_chunk_occurrences(baseline_cache) if native_evidence else None
    hyper_traces = []
    hyper_reused = 0
    for item in questions:
        checkpoint = _load_checkpoint_trace(checkpoint_root, "Hyper-RAG", item, sentence_ids, manifest_digest, split_digest, checkpoint_protocol)
        if checkpoint is not None:
            hyper_traces.append(checkpoint)
            hyper_reused += 1
            continue
        try:
            trace = await _hyper_trace_for_query(
                rag, operate, query_param, item["question_id"], item["question"], mapper,
                matched_source_token_budget, capture_final_context, native_evidence, chunk_occurrences,
            )
            _checkpoint_trace(checkpoint_root, "Hyper-RAG", item, trace, sentence_ids, manifest_digest, split_digest, checkpoint_protocol)
            hyper_traces.append(trace)
        except ProvenanceError as error:
            raise ProvenanceError(f"{item['question_id']}: {error}") from error
    cache_after = {"casc": _tree_digest(casc_cache), "hyper": _tree_digest(baseline_cache)}
    if cache_before != cache_after:
        raise RuntimeError("A persisted retrieval index changed during trace replay; outputs were not published")
    question_ids = {item["question_id"] for item in questions}
    if capture_final_context:
        errors = validate_final_context_traces(casc_traces, sentence_ids, question_ids, "CascHyper-RAG")
        errors.extend(validate_final_context_traces(hyper_traces, sentence_ids, question_ids, "Hyper-RAG"))
        if errors:
            raise ValueError("Final-context trace validation failed before publication:\n- " + "\n- ".join(errors))
    if native_evidence:
        errors = validate_native_evidence_traces(casc_traces, sentence_ids, question_ids, "CascHyper-RAG")
        errors.extend(validate_native_evidence_traces(hyper_traces, sentence_ids, question_ids, "Hyper-RAG"))
        if errors:
            raise ValueError("Native-evidence trace validation failed before publication:\n- " + "\n- ".join(errors))
    _publish_traces(casc_output, hyper_output, casc_traces, hyper_traces, sentence_ids, question_ids)
    return {"questions": len(questions), "casc_traces": len(casc_traces), "hyper_traces": len(hyper_traces), "casc_reused_checkpoints": casc_reused, "hyper_reused_checkpoints": hyper_reused}


async def _replay_matched_final_context_traces(
    manifest_path: str | Path,
    split_path: str | Path,
    contexts_path: str | Path,
    hyperrag_root: str | Path,
    casc_output: str | Path,
    hyper_output: str | Path,
    checkpoint_directory: str | Path,
    token_budget: int,
) -> dict[str, int]:
    """Run the controlled final-source-evidence RQ6 protocol without mutating indexes."""
    manifest, split, contexts = read_json(manifest_path), read_json(split_path), read_json(contexts_path)
    mapper = CanonicalSentenceMapper(manifest, contexts)
    questions, root = _question_items(split), Path(hyperrag_root)
    checkpoint_root = Path(checkpoint_directory)
    sentence_ids = {item["sentence_id"] for item in manifest["sentences"]}
    manifest_digest, split_digest = sha256_file(manifest_path), sha256_file(split_path)
    checkpoint_protocol = f"{MATCHED_FINAL_CONTEXT_PROTOCOL}:{token_budget}"
    runtime_dir = Path(__file__).resolve().parents[1] / ".rq6_runtime"
    runtime_dir.mkdir(exist_ok=True)
    casc_engine, casc_cache = _load_v81_engine(root, contexts, runtime_dir)
    baseline_cache = root / "Hyper-RAG-main" / "caches" / "deepseek-v4-flash" / "physics" / "index"
    cache_before = {"casc": _tree_digest(casc_cache), "hyper": _tree_digest(baseline_cache)}
    chunk_cache = _precompute_engine_chunk_provenance(casc_engine, mapper)
    casc_traces: list[dict[str, Any]] = []
    casc_reused = 0
    for item in questions:
        checkpoint = _load_checkpoint_trace(
            checkpoint_root, "CascHyper-RAG", item, sentence_ids, manifest_digest, split_digest, checkpoint_protocol
        )
        if checkpoint is not None:
            casc_traces.append(checkpoint)
            casc_reused += 1
            continue
        try:
            trace = await _matched_final_casc_trace(
                casc_engine, item["question_id"], item["question"], mapper, chunk_cache, token_budget
            )
            _checkpoint_trace(
                checkpoint_root, "CascHyper-RAG", item, trace, sentence_ids, manifest_digest, split_digest, checkpoint_protocol
            )
            casc_traces.append(trace)
        except ProvenanceError as error:
            raise ProvenanceError(f"{item['question_id']}: {error}") from error

    rag, query_param, operate = _load_hyperrag_main(
        root,
        runtime_dir,
        source_token_budget=native_retrieval_config("Hyper-RAG")["max_token_for_text_unit"],
        relation_context_budget=native_retrieval_config("Hyper-RAG")["max_token_for_relation_context"],
    )
    chunk_occurrences = _baseline_chunk_occurrences(baseline_cache)
    hyper_traces: list[dict[str, Any]] = []
    hyper_reused = 0
    for item in questions:
        checkpoint = _load_checkpoint_trace(
            checkpoint_root, "Hyper-RAG", item, sentence_ids, manifest_digest, split_digest, checkpoint_protocol
        )
        if checkpoint is not None:
            hyper_traces.append(checkpoint)
            hyper_reused += 1
            continue
        try:
            trace = await _matched_final_hyper_trace(
                rag, operate, item["question_id"], item["question"], query_param, mapper, chunk_occurrences, token_budget
            )
            _checkpoint_trace(
                checkpoint_root, "Hyper-RAG", item, trace, sentence_ids, manifest_digest, split_digest, checkpoint_protocol
            )
            hyper_traces.append(trace)
        except ProvenanceError as error:
            raise ProvenanceError(f"{item['question_id']}: {error}") from error
    cache_after = {"casc": _tree_digest(casc_cache), "hyper": _tree_digest(baseline_cache)}
    if cache_before != cache_after:
        raise RuntimeError("A persisted retrieval index changed during matched final-context replay; outputs were not published")
    question_ids = {item["question_id"] for item in questions}
    errors = validate_matched_final_context_traces(casc_traces, sentence_ids, question_ids, "CascHyper-RAG", token_budget)
    errors.extend(validate_matched_final_context_traces(hyper_traces, sentence_ids, question_ids, "Hyper-RAG", token_budget))
    if errors:
        raise ValueError("Matched final-context trace validation failed before publication:\n- " + "\n- ".join(errors))
    _publish_traces(casc_output, hyper_output, casc_traces, hyper_traces, sentence_ids, question_ids)
    return {
        "questions": len(questions),
        "casc_traces": len(casc_traces),
        "hyper_traces": len(hyper_traces),
        "casc_reused_checkpoints": casc_reused,
        "hyper_reused_checkpoints": hyper_reused,
    }


def replay_traces(manifest_path: str | Path, split_path: str | Path, contexts_path: str | Path, hyperrag_root: str | Path, casc_output: str | Path, hyper_output: str | Path, checkpoint_directory: str | Path = "data/checkpoints") -> dict[str, int]:
    """Run one complete trace replay in a single event loop."""
    return asyncio.run(_replay_traces(manifest_path, split_path, contexts_path, hyperrag_root, casc_output, hyper_output, checkpoint_directory))


def replay_final_context_traces(manifest_path: str | Path, split_path: str | Path, contexts_path: str | Path, hyperrag_root: str | Path, casc_output: str | Path, hyper_output: str | Path, checkpoint_directory: str | Path = "data/checkpoints_final_context") -> dict[str, int]:
    """Replay both native pipelines and preserve their generator-visible source spans."""
    return asyncio.run(_replay_traces(
        manifest_path, split_path, contexts_path, hyperrag_root, casc_output, hyper_output,
        checkpoint_directory, capture_final_context=True,
    ))


def replay_matched_final_context_traces(
    manifest_path: str | Path,
    split_path: str | Path,
    contexts_path: str | Path,
    hyperrag_root: str | Path,
    casc_output: str | Path,
    hyper_output: str | Path,
    token_budget: int = MATCHED_FINAL_SOURCE_TOKEN_BUDGET,
    checkpoint_directory: str | Path = "data/checkpoints_matched_final_context_12000",
) -> dict[str, int]:
    """Replay both native retrieval structures under one 12k final-source budget."""
    if token_budget != MATCHED_FINAL_SOURCE_TOKEN_BUDGET:
        raise ValueError("Matched final-context RQ6 protocol is frozen at 12000 source-evidence tokens")
    return asyncio.run(_replay_matched_final_context_traces(
        manifest_path, split_path, contexts_path, hyperrag_root, casc_output, hyper_output,
        checkpoint_directory, token_budget,
    ))


def replay_native_evidence_traces(manifest_path: str | Path, split_path: str | Path, contexts_path: str | Path, hyperrag_root: str | Path, casc_output: str | Path, hyper_output: str | Path, checkpoint_directory: str | Path = "data/checkpoints_native_evidence") -> dict[str, int]:
    """Replay both systems using their unchanged RQ1/RQ2 retrieval configurations."""
    return asyncio.run(_replay_traces(
        manifest_path, split_path, contexts_path, hyperrag_root, casc_output, hyper_output,
        checkpoint_directory, native_evidence=True,
    ))


def build_latent_topic_manifest(
    manifest_path: str | Path,
    contexts_path: str | Path,
    hyperrag_root: str | Path,
    output_path: str | Path,
) -> dict[str, int]:
    """Export the fixed v8.1 latent-topic candidate space without querying the RAG."""
    destination = Path(output_path)
    if destination.exists():
        raise ValueError(f"Latent topic manifest already exists and will not be overwritten: {destination}")
    temporary = destination.with_suffix(".json.tmp")
    if temporary.exists():
        raise ValueError(f"Temporary topic manifest path already exists: {temporary}")
    manifest, contexts = read_json(manifest_path), read_json(contexts_path)
    mapper = CanonicalSentenceMapper(manifest, contexts)
    runtime_dir = Path(__file__).resolve().parents[1] / ".rq6_runtime"
    runtime_dir.mkdir(exist_ok=True)
    engine, casc_cache = _load_v81_engine(Path(hyperrag_root), contexts, runtime_dir)
    cache_before = _tree_digest(casc_cache)
    topic_manifest = _latent_topic_manifest(
        engine, mapper, _precompute_engine_chunk_provenance(engine, mapper)
    )
    if cache_before != _tree_digest(casc_cache):
        raise RuntimeError("The persisted CascHyper-RAG index changed while creating the topic manifest")
    errors = validate_latent_topic_manifest(
        topic_manifest, {str(item["sentence_id"]) for item in manifest["sentences"]}
    )
    if errors:
        raise ValueError("Latent topic manifest validation failed:\n- " + "\n- ".join(errors))
    try:
        write_json(temporary, topic_manifest)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return {
        "topics": len(topic_manifest["topics"]),
        "chunks": int(topic_manifest["total_chunk_count"]),
    }


def replay_matched_traces(manifest_path: str | Path, split_path: str | Path, contexts_path: str | Path, hyperrag_root: str | Path, casc_output: str | Path, hyper_output: str | Path, source_token_budget: int = 6000, checkpoint_directory: str | Path = "data/checkpoints_matched_6000") -> dict[str, int]:
    """Replay both indexes under one shared source-text budget without changing either index."""
    if source_token_budget != 6000:
        raise ValueError("Matched RQ6 protocol is preregistered at 6000 source tokens (5 × 1200-token Casc chunks)")
    return asyncio.run(_replay_traces(manifest_path, split_path, contexts_path, hyperrag_root, casc_output, hyper_output, checkpoint_directory, source_token_budget))


def replay_casc_trace(manifest_path: str | Path, split_path: str | Path, contexts_path: str | Path, hyperrag_root: str | Path, casc_output: str | Path, checkpoint_directory: str | Path = "data/checkpoints") -> dict[str, int]:
    """Re-export a corrected CascHyper-RAG trace without replaying Hyper-RAG."""
    return asyncio.run(_replay_casc_trace(manifest_path, split_path, contexts_path, hyperrag_root, casc_output, checkpoint_directory))


def replay_matched_casc_trace(manifest_path: str | Path, split_path: str | Path, contexts_path: str | Path, hyperrag_root: str | Path, casc_output: str | Path, source_token_budget: int = 6000, checkpoint_directory: str | Path = "data/checkpoints_matched_6000_verified") -> dict[str, int]:
    """Re-export only the Casc trace under the strict shared source-text cap."""
    if source_token_budget != 6000:
        raise ValueError("Matched RQ6 protocol is preregistered at 6000 source tokens (5 × 1200-token Casc chunks)")
    return asyncio.run(_replay_casc_trace(manifest_path, split_path, contexts_path, hyperrag_root, casc_output, checkpoint_directory, source_token_budget))
