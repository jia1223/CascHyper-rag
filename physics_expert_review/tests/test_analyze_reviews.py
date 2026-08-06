import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
ANALYZE_SCRIPT = PACKAGE_DIR / "analyze_reviews.py"


class AnalyzeReviewsCliTests(unittest.TestCase):
    def test_unblinds_scores_and_writes_paired_statistics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            forms_dir = workspace / "forms"
            forms_dir.mkdir()
            key_path = workspace / "unblinding_key.csv"
            output_dir = workspace / "analysis"
            fields = [
                "question_id", "expert_id", "system_a_correctness", "system_a_completeness",
                "system_a_clarity", "system_b_correctness", "system_b_completeness",
                "system_b_clarity", "overall_preference",
            ]
            form_rows = {
                "expert_01": [
                    ["P001", "expert_01", "5", "4", "5", "3", "3", "3", "A"],
                    ["P002", "expert_01", "2", "2", "3", "4", "4", "4", "B"],
                ],
                "expert_02": [
                    ["P001", "expert_02", "3", "3", "3", "5", "4", "5", "B"],
                    ["P002", "expert_02", "4", "4", "4", "2", "2", "3", "A"],
                ],
            }
            for expert_id, rows in form_rows.items():
                with (forms_dir / f"{expert_id}.csv").open("w", encoding="utf-8-sig", newline="") as target:
                    writer = csv.writer(target)
                    writer.writerow(fields)
                    writer.writerows(rows)

            with key_path.open("w", encoding="utf-8-sig", newline="") as target:
                writer = csv.DictWriter(target, fieldnames=["question_id", "expert_id", "is_calibration", "system_a_method", "system_b_method", "stage"])
                writer.writeheader()
                writer.writerows(
                    [
                        {"question_id": "P001", "expert_id": "expert_01", "is_calibration": "false", "system_a_method": "casc", "system_b_method": "hyper", "stage": "unlabeled"},
                        {"question_id": "P002", "expert_id": "expert_01", "is_calibration": "false", "system_a_method": "hyper", "system_b_method": "casc", "stage": "unlabeled"},
                        {"question_id": "P001", "expert_id": "expert_02", "is_calibration": "false", "system_a_method": "hyper", "system_b_method": "casc", "stage": "unlabeled"},
                        {"question_id": "P002", "expert_id": "expert_02", "is_calibration": "false", "system_a_method": "casc", "system_b_method": "hyper", "stage": "unlabeled"},
                    ]
                )

            completed = subprocess.run(
                [sys.executable, str(ANALYZE_SCRIPT), "--forms-dir", str(forms_dir), "--unblinding-key", str(key_path), "--output-dir", str(output_dir), "--bootstrap-samples", "200", "--seed", "9"],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            summary = json.loads((output_dir / "summary.json").read_text(encoding="utf-8"))
            self.assertAlmostEqual(summary["metrics"]["correctness"]["difference_casc_minus_hyper"], 2.0)
            self.assertEqual(summary["preference_votes"], {"casc_win": 4, "tie": 0, "hyper_win": 0})
            self.assertIn("CascHyper-RAG", (output_dir / "report.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
