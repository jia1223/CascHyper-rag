"""Frozen RQ1/RQ2 retrieval settings used by the native-evidence RQ6 protocol."""

from __future__ import annotations


NATIVE_EVIDENCE_PROTOCOL = "rq6_rq1_rq2_native_selected_evidence_v1"

_CASC_CONFIG = {
    "top_k_chunks": 5,
    "top_k_sents_per_hop": 10,
    "enable_multi_hop": True,
    "consistency_verification": False,
}
_HYPER_CONFIG = {
    "mode": "hyper",
    "top_k_per_track": 10,
    "max_token_for_text_unit": 1200,
    "max_token_for_relation_context": 1600,
}


def native_retrieval_config(method: str) -> dict[str, int | str | bool]:
    """Return a copy of the immutable RQ1/RQ2 configuration for one method."""
    if method == "CascHyper-RAG":
        return dict(_CASC_CONFIG)
    if method == "Hyper-RAG":
        return dict(_HYPER_CONFIG)
    raise ValueError(f"Unknown native-evidence method: {method}")
