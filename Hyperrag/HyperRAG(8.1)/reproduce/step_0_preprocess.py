"""
Step 0: 数据预处理
从原始 JSONL 数据集中提取去重的 context 字段，保存为 JSON 数组。
"""
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from my_config import DATASETS_DIR, CACHES_DIR


def extract_unique_contexts(input_directory, output_directory):
    in_dir, out_dir = Path(input_directory), Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    jsonl_files = list(in_dir.glob("*.jsonl"))
    print(f"Found {len(jsonl_files)} JSONL files.")

    for file_path in jsonl_files:
        output_path = out_dir / f"{file_path.stem}_unique_contexts.json"
        if output_path.exists():
            print(f"[Skip] {output_path.name} already exists.")
            continue

        unique_contexts_dict = {}
        print(f"Processing file: {file_path.name}")

        try:
            with open(file_path, "r", encoding="utf-8") as infile:
                for line_number, line in enumerate(infile, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        json_obj = json.loads(line)
                        context = json_obj.get("context")
                        if context and context not in unique_contexts_dict:
                            unique_contexts_dict[context] = None
                    except json.JSONDecodeError as e:
                        print(f"  JSON error at line {line_number}: {e}")
        except FileNotFoundError:
            print(f"  File not found: {file_path.name}")
            continue
        except Exception as e:
            print(f"  Error: {e}")
            continue

        unique_contexts_list = list(unique_contexts_dict.keys())
        print(f"  -> {len(unique_contexts_list)} unique contexts")

        with open(output_path, "w", encoding="utf-8") as outfile:
            json.dump(unique_contexts_list, outfile, ensure_ascii=False, indent=4)
        print(f"  -> Saved to {output_path.name}")

    print("All files processed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 0: Extract unique contexts from JSONL datasets")
    parser.add_argument("-d", "--data_name", type=str, default="physics",
                        help="Dataset name (subdirectory under datasets/)")
    args = parser.parse_args()

    data_name = args.data_name
    input_dir = str(Path(DATASETS_DIR) / data_name)
    output_dir = str(Path(CACHES_DIR) / data_name / "contexts")

    print(f"{'='*60}")
    print(f"Step 0: Preprocessing dataset '{data_name}'")
    print(f"  Input:  {input_dir}")
    print(f"  Output: {output_dir}")
    print(f"{'='*60}")

    extract_unique_contexts(input_dir, output_dir)
