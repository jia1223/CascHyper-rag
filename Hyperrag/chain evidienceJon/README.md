# RQ6 Physics Evidence-Chain Evaluation

This directory is an isolated evaluation package for RQ6.  It **does not**
modify or import the existing RAG implementations, their caches, or their
answer-generation outputs.

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

## 3. Retrieval-trace contract

Each method supplies one JSON list following
`templates/retrieval_trace.template.json`.  Every retrieved unit must map to
canonical `sentence_id` values from `corpus_manifest.json`.

`retrieved_bridge_entities` is descriptive only. A scored bridge must be
exported in `retrieved_bridges` with its rank, canonical entity, and the two
canonical sentence endpoints. A bridge is credited only if that edge matches
the gold relation and both adjacent gold hops are supported within the relevant
top-k evidence set.

`selected_topic_ids` is optional.  It is intended for CascHyper-RAG's
topic-routing diagnostic.  Hyper-RAG has no equivalent topic-routing module,
so topic-routing coverage is reported as `null` rather than forced into an
unfair comparison.

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
- Topic-routing Coverage: CascHyper-RAG-only diagnostic based on selected topic
  IDs and the gold topic set.

All between-method differences are paired and use bootstrap confidence
intervals.  No answer quality score is computed here.
