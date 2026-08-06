import csv
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = PACKAGE_DIR / "prepare_review.py"


class PrepareReviewCliTests(unittest.TestCase):
    def test_creates_blinded_expert_forms_and_private_unblinding_key(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            ours_path = workspace / "casc.json"
            baseline_path = workspace / "hyper.json"
            references_path = workspace / "references.json"
            metadata_path = workspace / "metadata.json"
            output_dir = workspace / "review"

            questions = [
                {"query": "What is question one?", "result": "Casc answer one."},
                {"query": "What is question two?", "result": "Casc answer two."},
                {"query": "What is question three?", "result": "Casc answer three."},
            ]
            ours_path.write_text(json.dumps(questions), encoding="utf-8")
            baseline_path.write_text(
                json.dumps(
                    [
                        {"query": item["query"], "result": item["result"].replace("Casc", "Hyper")}
                        for item in questions
                    ]
                ),
                encoding="utf-8",
            )
            references_path.write_text(
                json.dumps(
                    {item["query"]: f"Reference for {index}" for index, item in enumerate(questions, start=1)}
                ),
                encoding="utf-8",
            )
            metadata_path.write_text(
                json.dumps(
                    {
                        method: {
                            "code_version": "test",
                            "generation_model": "test-model",
                            "generation_prompt_version": "v1",
                            "decoding_parameters": "temperature=0",
                            "context_token_budget": "1024",
                            "embedding_model": "test-embedding",
                        }
                        for method in ("casc_hyper_rag", "hyper_rag")
                    }
                ),
                encoding="utf-8",
            )

            completed = subprocess.run(
                [
                    sys.executable,
                    str(PREPARE_SCRIPT),
                    "--casc-results",
                    str(ours_path),
                    "--hyper-results",
                    str(baseline_path),
                    "--references",
                    str(references_path),
                    "--run-metadata",
                    str(metadata_path),
                    "--output-dir",
                    str(output_dir),
                    "--sample-size",
                    "2",
                    "--expert-count",
                    "2",
                    "--calibration-size",
                    "0",
                    "--seed",
                    "17",
                ],
                check=False,
                text=True,
                capture_output=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            form_paths = sorted((output_dir / "materials").glob("expert_*.csv"))
            self.assertEqual(len(form_paths), 2)

            with form_paths[0].open(encoding="utf-8-sig", newline="") as source:
                rows = list(csv.DictReader(source))
            self.assertEqual(len(rows), 2)
            self.assertTrue(all(row["first_system_answer"] for row in rows))
            self.assertTrue(all(row["second_system_answer"] for row in rows))
            self.assertTrue(all(row["reference_answer"] for row in rows))
            all_display_orders = set()
            for form_path in form_paths:
                with form_path.open(encoding="utf-8-sig", newline="") as source:
                    all_display_orders.update(row["display_order"] for row in csv.DictReader(source))
            self.assertEqual(all_display_orders, {"A/B", "B/A"})
            self.assertNotIn("CascHyper-RAG", form_paths[0].read_text(encoding="utf-8-sig"))
            self.assertNotIn("Hyper-RAG", form_paths[0].read_text(encoding="utf-8-sig"))

            key_path = output_dir / "private" / "unblinding_key.csv"
            self.assertTrue(key_path.exists())
            with key_path.open(encoding="utf-8-sig", newline="") as source:
                key_rows = list(csv.DictReader(source))
            self.assertEqual(len(key_rows), 4)
            self.assertEqual({row["system_a_method"] for row in key_rows} | {row["system_b_method"] for row in key_rows}, {"casc", "hyper"})


if __name__ == "__main__":
    unittest.main()
