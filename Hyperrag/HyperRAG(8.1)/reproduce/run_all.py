"""
Run All: 一键运行实验流程
支持指定数据集、问题阶段和各步骤的开关。
"""
import sys
import argparse
import subprocess
from pathlib import Path

REPRODUCE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def run_step(script_name, args_list, description):
    """运行单个步骤脚本"""
    script_path = REPRODUCE_DIR / script_name
    cmd = [PYTHON, str(script_path)] + args_list

    print(f"\n{'#'*60}")
    print(f"# {description}")
    print(f"# Command: {' '.join(cmd)}")
    print(f"{'#'*60}\n")

    result = subprocess.run(cmd, cwd=str(REPRODUCE_DIR.parent))
    if result.returncode != 0:
        print(f"\n[Error] {script_name} failed with return code {result.returncode}")
        return False
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run all experiment steps for HyperRAG v8.1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all steps for 'mix' dataset, 2-stage questions
  python run_all.py -d mix -s 2

  # Run only preprocessing and index building
  python run_all.py -d mix --steps 0 1

  # Run question generation with 50 questions
  python run_all.py -d mix -s 2 -n 50 --steps 2

  # Run scoring evaluation only
  python run_all.py -d mix -s 2 --steps 4

  # Run all 9 datasets
  python run_all.py --all-datasets -s 2
        """
    )
    parser.add_argument("-d", "--data_name", type=str, default="mix",
                        help="Dataset name")
    parser.add_argument("-s", "--stage", type=int, default=2, choices=[1, 2, 3],
                        help="Question stage")
    parser.add_argument("-n", "--num_questions", type=int, default=None,
                        help="Number of questions (overrides config)")
    parser.add_argument("--steps", nargs="+", type=int, default=None,
                        help="Specific steps to run (e.g. --steps 0 1 2)")
    parser.add_argument("--all-datasets", action="store_true",
                        help="Run on all 9 datasets")
    parser.add_argument("--mode-b", type=str, default=None,
                        help="Comparison mode for step 5 selection evaluation")
    args = parser.parse_args()

    ALL_DATASETS = [
        "agriculture", "art", "fin", "legal",
        "mathematics", "mix", "neurology", "pathology", "physics"
    ]

    datasets = ALL_DATASETS if args.all_datasets else [args.data_name]
    steps_to_run = set(args.steps) if args.steps else {0, 1, 2, 3, 4}

    print(f"{'='*60}")
    print(f"HyperRAG v8.1 Experiment Runner")
    print(f"  Datasets: {datasets}")
    print(f"  Stage:    {args.stage}")
    print(f"  Steps:    {sorted(steps_to_run)}")
    print(f"{'='*60}")

    for data_name in datasets:
        print(f"\n{'*'*60}")
        print(f"* Dataset: {data_name}")
        print(f"{'*'*60}")

        common_args = ["-d", data_name, "-s", str(args.stage)]

        # Step 0: Preprocess
        if 0 in steps_to_run:
            success = run_step(
                "step_0_preprocess.py",
                ["-d", data_name],
                f"Step 0: Preprocessing '{data_name}'"
            )
            if not success:
                continue

        # Step 1: Build Index
        if 1 in steps_to_run:
            success = run_step(
                "step_1_build_index.py",
                ["-d", data_name],
                f"Step 1: Building index for '{data_name}'"
            )
            if not success:
                continue

        # Step 2: Generate Questions
        if 2 in steps_to_run:
            step2_args = common_args.copy()
            if args.num_questions:
                step2_args += ["-n", str(args.num_questions)]
            success = run_step(
                "step_2_extract_question.py",
                step2_args,
                f"Step 2: Generating {args.stage}-stage questions for '{data_name}'"
            )
            if not success:
                continue

        # Step 3: Generate Answers
        if 3 in steps_to_run:
            success = run_step(
                "step_3_response_question.py",
                common_args,
                f"Step 3: Generating v8.1 answers for '{data_name}'"
            )
            if not success:
                continue

        # Step 4: Scoring Evaluation
        if 4 in steps_to_run:
            success = run_step(
                "step_4_evaluate_scoring.py",
                common_args + ["-m", "v81"],
                f"Step 4: Scoring evaluation for '{data_name}'"
            )

        # Step 5: Selection Evaluation (optional, needs two answer sets)
        if 5 in steps_to_run:
            mode_b = args.mode_b or "naive"
            success = run_step(
                "step_5_evaluate_selection.py",
                common_args + ["-a", "v81", "-b", mode_b],
                f"Step 5: Selection evaluation (v81 vs {mode_b}) for '{data_name}'"
            )

    print(f"\n{'='*60}")
    print("All experiments completed!")
    print(f"{'='*60}")
