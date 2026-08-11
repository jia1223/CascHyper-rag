"""Command-line entry points for the isolated RQ6 evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluate import evaluate_pair
from .io_utils import read_json, write_json
from .plotting import create_plots
from .prepare import prepare_inputs
from .candidates import generate_annotation_packages
from .adjudication import generate_adjudication_queue
from .merge import merge_adjudications
from .trace_replay import replay_casc_trace, replay_matched_casc_trace, replay_matched_traces, replay_traces
from .validation import validate_gold, validate_question_split, validate_traces


def _prepare(args: argparse.Namespace) -> int:
    manifest = prepare_inputs(args.contexts, args.questions_root, args.output, args.stage1_count, args.seed)
    print(f"Prepared {manifest['counts']['questions']} questions and {manifest['counts']['sentences']} canonical sentences.")
    return 0


def _validate_gold(args: argparse.Namespace) -> int:
    gold = read_json(args.gold)
    sentences = read_json(args.sentences)["sentences"]
    errors = validate_gold(gold, {item["sentence_id"] for item in sentences})
    errors.extend(validate_question_split(gold, read_json(args.split)))
    if errors:
        print("Gold validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    print(f"Gold validation passed for {len(gold)} questions.")
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    gold = read_json(args.gold)
    split = read_json(args.split)
    sentences = read_json(args.sentences)["sentences"]
    sentence_ids = {item["sentence_id"] for item in sentences}
    casc_traces = read_json(args.casc_trace)
    hyper_traces = read_json(args.hyper_trace)
    errors = validate_gold(gold, sentence_ids)
    errors.extend(validate_question_split(gold, split))
    expected_question_ids = {item["question_id"] for item in split["items"]}
    errors.extend(validate_traces(casc_traces, sentence_ids, expected_question_ids, "CascHyper-RAG"))
    errors.extend(validate_traces(hyper_traces, sentence_ids, expected_question_ids, "Hyper-RAG"))
    if errors:
        print("RQ6 input validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1
    results = evaluate_pair(
        gold_items=gold,
        casc_traces=casc_traces,
        hyper_traces=hyper_traces,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    output = Path(args.output)
    write_json(output / "evaluation_results.json", results)
    if args.plots:
        paths = create_plots(results, output)
        print("Created figures: " + ", ".join(str(path) for path in paths))
    print(f"Evaluation written to {output / 'evaluation_results.json'}")
    return 0


def _generate_candidates(args: argparse.Namespace) -> int:
    manifest = generate_annotation_packages(
        manifest_path=args.manifest,
        split_path=args.split,
        questions_root=args.questions_root,
        output_directory=args.output,
        top_k=args.top_k,
    )
    print(f"Generated A/B annotation packages for {manifest['question_count']} questions.")
    return 0


def _generate_adjudication(args: argparse.Namespace) -> int:
    manifest = generate_adjudication_queue(
        manifest_path=args.manifest,
        split_path=args.split,
        annotator_a_directory=args.annotator_a,
        annotator_b_directory=args.annotator_b,
        output_directory=args.output,
    )
    print(f"Generated adjudication queue for {manifest['disputed_question_count']} disputed questions.")
    return 0


def _merge_adjudication(args: argparse.Namespace) -> int:
    summary = merge_adjudications(
        manifest_path=args.manifest,
        split_path=args.split,
        annotator_a_directory=args.annotator_a,
        annotator_b_directory=args.annotator_b,
        decisions_directory=args.decisions,
        output_path=args.output,
    )
    print(f"Frozen gold written to {args.output} ({summary}).")
    return 0


def _replay_traces(args: argparse.Namespace) -> int:
    summary = replay_traces(
        manifest_path=args.manifest,
        split_path=args.split,
        contexts_path=args.contexts,
        hyperrag_root=args.hyperrag_root,
        casc_output=args.casc_output,
        hyper_output=args.hyper_output,
        checkpoint_directory=args.checkpoint_dir,
    )
    print(f"Trace replay completed: {summary}.")
    return 0


def _replay_casc_trace(args: argparse.Namespace) -> int:
    summary = replay_casc_trace(
        manifest_path=args.manifest,
        split_path=args.split,
        contexts_path=args.contexts,
        hyperrag_root=args.hyperrag_root,
        casc_output=args.casc_output,
        checkpoint_directory=args.checkpoint_dir,
    )
    print(f"CascHyper-RAG trace replay completed: {summary}.")
    return 0


def _replay_matched_traces(args: argparse.Namespace) -> int:
    summary = replay_matched_traces(
        manifest_path=args.manifest,
        split_path=args.split,
        contexts_path=args.contexts,
        hyperrag_root=args.hyperrag_root,
        casc_output=args.casc_output,
        hyper_output=args.hyper_output,
        source_token_budget=args.token_budget,
        checkpoint_directory=args.checkpoint_dir,
    )
    print(f"Matched-budget trace replay completed: {summary}.")
    return 0


def _replay_matched_casc_trace(args: argparse.Namespace) -> int:
    summary = replay_matched_casc_trace(
        manifest_path=args.manifest,
        split_path=args.split,
        contexts_path=args.contexts,
        hyperrag_root=args.hyperrag_root,
        casc_output=args.casc_output,
        source_token_budget=args.token_budget,
        checkpoint_directory=args.checkpoint_dir,
    )
    print(f"Matched-budget CascHyper-RAG trace replay completed: {summary}.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Isolated RQ6 Physics evidence-chain evaluator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="Create canonical corpus and fixed question manifests")
    prepare.add_argument("--contexts", required=True)
    prepare.add_argument("--questions-root", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--stage1-count", type=int, default=20)
    prepare.add_argument("--seed", type=int, default=20260809)
    prepare.set_defaults(handler=_prepare)
    validate = subparsers.add_parser("validate-gold", help="Validate adjudicated gold evidence chains")
    validate.add_argument("--gold", required=True)
    validate.add_argument("--sentences", required=True)
    validate.add_argument("--split", required=True)
    validate.set_defaults(handler=_validate_gold)
    evaluate = subparsers.add_parser("evaluate", help="Evaluate paired CascHyper-RAG and Hyper-RAG traces")
    evaluate.add_argument("--gold", required=True)
    evaluate.add_argument("--sentences", required=True)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--casc-trace", required=True)
    evaluate.add_argument("--hyper-trace", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.add_argument("--bootstrap-samples", type=int, default=10000)
    evaluate.add_argument("--seed", type=int, default=20260809)
    evaluate.add_argument("--plots", action="store_true")
    evaluate.set_defaults(handler=_evaluate)
    candidates = subparsers.add_parser("generate-candidates", help="Create independent A/B gold-annotation candidate packs")
    candidates.add_argument("--manifest", required=True)
    candidates.add_argument("--split", required=True)
    candidates.add_argument("--questions-root", required=True)
    candidates.add_argument("--output", required=True)
    candidates.add_argument("--top-k", type=int, default=40)
    candidates.set_defaults(handler=_generate_candidates)
    adjudication = subparsers.add_parser("generate-adjudication", help="Create decision packages for A/B annotation disagreements")
    adjudication.add_argument("--manifest", required=True)
    adjudication.add_argument("--split", required=True)
    adjudication.add_argument("--annotator-a", required=True)
    adjudication.add_argument("--annotator-b", required=True)
    adjudication.add_argument("--output", required=True)
    adjudication.set_defaults(handler=_generate_adjudication)
    merge = subparsers.add_parser("merge-adjudication", help="Merge completed A/B adjudications into frozen gold")
    merge.add_argument("--manifest", required=True)
    merge.add_argument("--split", required=True)
    merge.add_argument("--annotator-a", required=True)
    merge.add_argument("--annotator-b", required=True)
    merge.add_argument("--decisions", required=True)
    merge.add_argument("--output", required=True)
    merge.set_defaults(handler=_merge_adjudication)
    replay = subparsers.add_parser("replay-traces", help="Replay existing Physics indexes into canonical RQ6 traces")
    replay.add_argument("--manifest", required=True)
    replay.add_argument("--split", required=True)
    replay.add_argument("--contexts", required=True)
    replay.add_argument("--hyperrag-root", required=True)
    replay.add_argument("--casc-output", required=True)
    replay.add_argument("--hyper-output", required=True)
    replay.add_argument("--checkpoint-dir", default="data/checkpoints")
    replay.set_defaults(handler=_replay_traces)
    replay_casc = subparsers.add_parser("replay-casc-trace", help="Re-export only CascHyper-RAG from its existing Physics index")
    replay_casc.add_argument("--manifest", required=True)
    replay_casc.add_argument("--split", required=True)
    replay_casc.add_argument("--contexts", required=True)
    replay_casc.add_argument("--hyperrag-root", required=True)
    replay_casc.add_argument("--casc-output", required=True)
    replay_casc.add_argument("--checkpoint-dir", default="data/checkpoints")
    replay_casc.set_defaults(handler=_replay_casc_trace)
    matched = subparsers.add_parser("replay-matched-traces", help="Replay both methods with the preregistered shared 6000-token source-text budget")
    matched.add_argument("--manifest", required=True)
    matched.add_argument("--split", required=True)
    matched.add_argument("--contexts", required=True)
    matched.add_argument("--hyperrag-root", required=True)
    matched.add_argument("--casc-output", required=True)
    matched.add_argument("--hyper-output", required=True)
    matched.add_argument("--token-budget", type=int, default=6000)
    matched.add_argument("--checkpoint-dir", default="data/checkpoints_matched_6000")
    matched.set_defaults(handler=_replay_matched_traces)
    matched_casc = subparsers.add_parser("replay-matched-casc-trace", help="Re-export only CascHyper-RAG under the strict shared 6000-token cap")
    matched_casc.add_argument("--manifest", required=True)
    matched_casc.add_argument("--split", required=True)
    matched_casc.add_argument("--contexts", required=True)
    matched_casc.add_argument("--hyperrag-root", required=True)
    matched_casc.add_argument("--casc-output", required=True)
    matched_casc.add_argument("--token-budget", type=int, default=6000)
    matched_casc.add_argument("--checkpoint-dir", default="data/checkpoints_matched_6000_verified")
    matched_casc.set_defaults(handler=_replay_matched_casc_trace)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
