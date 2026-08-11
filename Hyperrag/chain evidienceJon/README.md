# RQ6 Physics Evidence-Chain Evaluation

This directory is an isolated evaluation package for RQ6. It does not modify
the existing RAG implementations, their persisted indexes, or their
answer-generation outputs; its replay adapter imports the two implementations
only to query their already-built Physics indexes.

It evaluates only two externally exported retrieval traces:

- `CascHyper-RAG`
- `Hyper-RAG`

The workflow is:

1. Create a frozen canonical Physics sentence manifest and a 20/50/50 question
   split from the existing cached corpus.
2. Annotate gold evidence chains using `templates/gold_evidence_chains.template.json`.
3. Export each method's retrieval output into the trace contract in
   `templates/retrieval_trace.template.json`.
4. Validate gold annotations and evaluate paired retrieval metrics.

## Installation

The evaluator uses only the Python standard library.  Figures additionally
require `matplotlib`.

```powershell
cd 'E:\研究生\研究生论文\HoloHyper_RAG_Double_kbs_\Hyperrag\chain evidienceJon'
python -m unittest discover -s tests -v
```

## 1. Prepare frozen inputs

```powershell
python -m rq6_evidence.cli prepare `
  --contexts '..\HyperRAG(8.1)\caches_v81\physics\contexts\physics_unique_contexts.json' `
  --questions-root '..\HyperRAG(8.1)\caches_v81\physics\questions' `
  --output data\prepared `
  --seed 20260809
```

This produces:

- `corpus_manifest.json`: canonical `document_id`, `sentence_id`, character
  spans, text, and a document-section proxy topic;
- `question_split.json`: 20 randomly sampled Stage-1 items plus all Stage-2
  and Stage-3 items;
- `run_manifest.json`: immutable preparation settings and source hashes.

The Stage-2 and Stage-3 labels are only sampling strata.  They must be checked
by annotators before being considered valid multi-hop chains.

## 2. Gold annotation contract

Copy `templates/gold_evidence_chains.template.json` to
`data/gold_evidence_chains.json`.  A gold item contains ordered indispensable
sentences, bridge entities, gold topics, and its eligibility decision.

`eligible_for_full_chain` must be `true` only when all required hops are
indispensable.  Stage 1 needs one hop, Stage 2 needs two hops and one bridge,
and Stage 3 needs three hops and two bridges.

## 2.1 Generate independent annotation candidate packs

```powershell
python -m rq6_evidence.cli generate-candidates `
  --manifest data\prepared\corpus_manifest.json `
  --split data\prepared\question_split.json `
  --questions-root '..\HyperRAG(8.1)\caches_v81\physics\questions' `
  --output data\annotation_packages `
  --top-k 40
```

The command produces 120 Markdown read-packages and 120 editable JSON drafts
for each of annotators A and B. Each read-package includes its question, the
aligned `stage_ref` candidate text, and 40 deterministically retrievable source
sentences; its paired file at `annotator_A/B/drafts/<question_id>.json` is the
only file the annotator edits. The candidate index is an aid, not gold evidence;
annotators may select any canonical sentence from the frozen manifest when a
needed sentence is absent. If a question cannot support a complete chain, set
`eligible_for_full_chain` to `false` and use empty `gold_hops` and `bridges`.
Detailed Chinese instructions for the two annotators are in
[`ANNOTATION_GUIDE_zh.md`](ANNOTATION_GUIDE_zh.md).

## 2.2 Generate an A/B adjudication queue

After both annotators have completed all drafts, create packages only for
questions that differ on eligibility, ordered gold hops, normalized bridges, or
gold topics:

```powershell
python -m rq6_evidence.cli generate-adjudication `
  --manifest data\prepared\corpus_manifest.json `
  --split data\prepared\question_split.json `
  --annotator-a data\annotation_packages\annotator_A\drafts `
  --annotator-b data\annotation_packages\annotator_B\drafts `
  --output data\adjudication
```

This preserves A/B originals and writes a Markdown review package plus one
editable decision JSON per disputed question. The adjudicator selects `use_A`,
`use_B`, `revised`, or `ineligible`; a later merge step will produce the single
frozen `data/gold_evidence_chains.json` used for retrieval evaluation.
For reproducibility, `--output` must be a new or empty directory; use a new
directory if A/B drafts are revised, so existing adjudication decisions remain
untouched.

## 2.3 Freeze the adjudicated gold file

After all decision files are complete, merge the 92 adjudicated records with
the 28 A/B-consistent records. The output path is write-once and is validated
against the frozen split and canonical sentence manifest before it is created.

```powershell
python -m rq6_evidence.cli merge-adjudication `
  --manifest data\prepared\corpus_manifest.json `
  --split data\prepared\question_split.json `
  --annotator-a data\annotation_packages\annotator_A\drafts `
  --annotator-b data\annotation_packages\annotator_B\drafts `
  --decisions data\adjudication\decisions `
  --output data\gold_evidence_chains.json
```

For an A/B-consistent question, annotator A's record is retained (A/B are
already identical on eligibility, ordered hops, bridge relations, and topics).
For a disputed question, the completed decision selects A, B, a revised
`final_gold`, or a valid empty-chain ineligible record.

## 3.1 Replay existing indexes into retrieval traces

The answer files do not retain retrieval provenance. This command replays the
already-built Physics indexes for CascHyper-RAG (`hyperrag_v81`) and Hyper-RAG
(`hyperrag_main`), maps their returned text to frozen canonical sentence IDs,
and writes immutable trace files. It does not rebuild either index.

```powershell
D:\miniconda\envs\hypergraphrag\python.exe -m rq6_evidence.cli replay-traces `
  --manifest data\prepared\corpus_manifest.json `
  --split data\prepared\question_split.json `
  --contexts '..\HyperRAG(8.1)\caches_v81\physics\contexts\physics_unique_contexts.json' `
  --hyperrag-root .. `
  --casc-output data\caschyperrag_trace.json `
  --hyper-output data\hyperrag_trace.json
```

The replay still performs query-time model calls for query expansion, keyword
extraction, and embeddings. It redirects CascHyper-RAG prompt responses to its
isolated runtime directory, disables Hyper-RAG's query cache, verifies hashes
of both persisted retrieval indexes before publication, and never rebuilds or
rewrites either index.

If a method selects no usable source text units for a question, the trace keeps
that question with `retrieval_status: "empty"` and empty retrieval lists. This
is a valid zero-recall outcome, not a failed experiment export.

Replay is resumable by default. After every completed question it atomically
writes `data/checkpoints/<method>/<question_id>.json`; a later command reuses
only checkpoints whose method, question/stage, frozen manifest hash, split hash,
and canonical-span validation all still match. Delete this directory only when
you intentionally want a complete replay from scratch.

If the CascHyper-RAG trace-export contract changes, re-export only that method
and keep the already-frozen Hyper-RAG trace unchanged. The Casc checkpoint
schema protects against reusing an older export format; publish to a new output
name so prior runs remain auditable.

```powershell
D:\miniconda\envs\hypergraphrag\python.exe -m rq6_evidence.cli replay-casc-trace `
  --manifest data\prepared\corpus_manifest.json `
  --split data\prepared\question_split.json `
  --contexts '..\HyperRAG(8.1)\caches_v81\physics\contexts\physics_unique_contexts.json' `
  --hyperrag-root .. `
  --casc-output data\caschyperrag_trace_bridge_v2.json
```

Hyper-RAG renders its final Sources field from an unordered set, so its native
output has no reproducible cross-track source rank. For fair fixed-budget RQ6
scoring, this package preregisters `rq6_track_merge_v1`: selected relation-track
text units first, then selected entity-track units; ties keep the original
within-track selection order. The trace is therefore an explicit, deterministic
evaluation ranking over Hyper-RAG's actual selected source set, not a claim that
the upstream CSV renderer supplies a global rank.
See [`ADJUDICATION_GUIDE_zh.md`](ADJUDICATION_GUIDE_zh.md) for the Chinese
adjudicator workflow and decision examples.

## 3.2 Matched source-text budget sensitivity analysis

The native replay is an end-to-end system comparison. To test whether its
outcome is explained by different amounts of retrieved source text, run the
separate `rq6_matched_source_text_budget_v1` protocol. It fixes the maximum
source-text budget at 6000 tokens: CascHyper-RAG uses its native five cached
chunks (each indexed with a 1200-token cap), while Hyper-RAG first produces its
native relation-then-entity ranking and then retains whole source units until a
single shared 6000-token cap is reached. It never rebuilds either index.

```powershell
D:\miniconda\envs\hypergraphrag\python.exe -m rq6_evidence.cli replay-matched-traces `
  --manifest data\prepared\corpus_manifest.json `
  --split data\prepared\question_split.json `
  --contexts '..\HyperRAG(8.1)\caches_v81\physics\contexts\physics_unique_contexts.json' `
  --hyperrag-root .. `
  --casc-output data\caschyperrag_trace_matched_6000_strict.json `
  --hyper-output data\hyperrag_trace_matched_6000_strict.json `
  --checkpoint-dir data\checkpoints_matched_6000_strict

D:\miniconda\envs\hypergraphrag\python.exe -m rq6_evidence.cli evaluate `
  --gold data\gold_evidence_chains.json `
  --sentences data\prepared\corpus_manifest.json `
  --split data\prepared\question_split.json `
  --casc-trace data\caschyperrag_trace_matched_6000_strict.json `
  --hyper-trace data\hyperrag_trace_matched_6000_strict.json `
  --output results_matched_6000_strict `
  --bootstrap-samples 10000 `
  --seed 20260809 `
  --plots
```

This is a shared-cap comparison, not a claim that every query consumes exactly
6000 tokens: a method may naturally return fewer complete source units. Report
the native and matched-cap results together.

### 3.3 Latent-topic candidate coverage

The v8.1 topic IDs are latent routing clusters, not document-heading labels.
Therefore RQ6 Topic Coverage is defined as the fraction of questions for which
**all** gold evidence sentences lie in **any fixed Casc chunk admitted by the
topics selected for that question**:

\[
\mathrm{TopicCandidateCoverage}_i =
\mathbb{I}[G_i \subseteq C(T_i)].
\]

Here, \(G_i\) is the frozen set of gold sentence IDs and \(C(T_i)\) is the
union of canonical sentence spans covered by every indexed Casc chunk whose
latent-topic membership intersects the query-selected topic set \(T_i\).  It
is a CascHyper-RAG-only routing diagnostic; Hyper-RAG correctly remains `null`.
Always report it with `topic_candidate_diagnostics.mean_candidate_reduction`,
because highly overlapping latent memberships can retain most of the corpus.

Create the immutable topic-to-source-span manifest once, then re-export only
the Casc trace (the existing Hyper trace and frozen gold are unchanged):

```powershell
D:\miniconda\envs\hypergraphrag\python.exe -m rq6_evidence.cli prepare-latent-topic-manifest `
  --manifest data\prepared\corpus_manifest.json `
  --contexts '..\HyperRAG(8.1)\caches_v81\physics\contexts\physics_unique_contexts.json' `
  --hyperrag-root .. `
  --output data\casc_latent_topic_manifest_v2.json

D:\miniconda\envs\hypergraphrag\python.exe -m rq6_evidence.cli replay-matched-casc-trace `
  --manifest data\prepared\corpus_manifest.json `
  --split data\prepared\question_split.json `
  --contexts '..\HyperRAG(8.1)\caches_v81\physics\contexts\physics_unique_contexts.json' `
  --hyperrag-root .. `
  --casc-output data\caschyperrag_trace_matched_6000_latent_topics.json `
  --checkpoint-dir data\checkpoints_matched_6000_latent_topics
```

For the already completed Physics run in this workspace, the strict Casc trace
is `data/caschyperrag_trace_matched_6000_verified.json`; it was paired with the
independently verified 6000-cap Hyper trace
`data/hyperrag_trace_matched_6000.json`. The resulting report is
`results_matched_6000_verified/evaluation_results.json`. Do not use the older
`caschyperrag_trace_matched_6000.json`, which predates strict per-query Casc
token accounting.

## 3. Retrieval-trace contract

Each method supplies one JSON list following
`templates/retrieval_trace.template.json`.  Every retrieved unit must map to
canonical `sentence_id` values from `corpus_manifest.json`.

`retrieved_bridge_entities` is descriptive only. A scored bridge must be
exported in `retrieved_bridges` with its rank, canonical entity, and the two
canonical sentence endpoints. RQ6 treats that entity-mediated connection as
**undirected**: endpoint order is ignored, while the entity and both sentence
IDs must match gold and both hops must be supported within the relevant top-k
evidence set.

`selected_topic_ids` is optional.  It is intended for CascHyper-RAG's
topic-candidate diagnostic.  In a latent-topic trace they use the fixed
`latent:<id>` namespace and must be evaluated with the accompanying immutable
topic manifest. Hyper-RAG has no equivalent topic-routing module, so its topic
coverage is reported as `null` rather than forced into an unfair comparison.

## 4. Validate and evaluate

```powershell
python -m rq6_evidence.cli validate-gold `
  --gold data\gold_evidence_chains.json `
  --sentences data\prepared\corpus_manifest.json `
  --split data\prepared\question_split.json

python -m rq6_evidence.cli evaluate `
  --gold data\gold_evidence_chains.json `
  --sentences data\prepared\corpus_manifest.json `
  --split data\prepared\question_split.json `
  --casc-trace data\caschyperrag_trace.json `
  --hyper-trace data\hyperrag_trace.json `
  --casc-topic-manifest data\casc_latent_topic_manifest_v2.json `
  --output results `
  --bootstrap-samples 10000 `
  --seed 20260809 `
  --plots
```

The evaluator writes `results/evaluation_results.json`, a CascHyper-RAG-only
topic-to-chain funnel, a paired evidence comparison, and a stage-level
full-chain figure when `matplotlib` is available.

## Metrics

- Chunk Recall@5: fraction of gold evidence sentences covered by the top five
  returned chunks.
- Sentence Recall@10: fraction of gold evidence sentences supported by the top
  ten returned evidence units.
- Bridge Entity Recall@10: fraction of gold bridges recovered with both
  adjacent gold hops supported.
- Full-chain Recall@5/@10/@20: all required gold hops and bridges are
  recovered within the stated evidence budget.
- Topic Candidate Coverage: CascHyper-RAG-only rate at which every gold
  evidence sentence falls in source chunks admitted by the selected latent
  topics. Report this jointly with candidate-chunk reduction, not as a
  standalone retrieval gain.

All between-method differences are paired and use bootstrap confidence
intervals. `evaluation_results.json` also reports sentence-provenance coverage.
An LLM-derived retrieval unit that cannot be grounded to an original-document
sentence is retained in trace diagnostics but excluded from sentence/bridge
credit; no answer quality score is computed here.
