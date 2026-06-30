"""
Step 5: Selection 评估
使用 GPT-4o-mini 将两组回答在 8 个维度上进行对比，选择胜者并统计胜率。
不需要参考答案。
"""
import re
import sys
import json
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from my_config import EVAL_LLM_API_KEY, EVAL_LLM_BASE_URL, EVAL_LLM_MODEL, CACHES_DIR


def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs) -> str:
    openai_client = OpenAI(api_key=EVAL_LLM_API_KEY, base_url=EVAL_LLM_BASE_URL)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})
    response = openai_client.chat.completions.create(
        model=EVAL_LLM_MODEL, messages=messages, **kwargs
    )
    return response.choices[0].message.content


def extract_queries_and_answers(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        query_list = json.load(f)
    queries = [i["query"] for i in query_list]
    answers = [i["result"] for i in query_list]
    return queries, answers


def exam_by_selection(queries, A_answers, B_answers):
    responses = []
    sys_prompt = """
        ---Role---
        You will evaluate two answers to the same question based on eight criteria: **Comprehensiveness**, **Empowerment**, **Accuracy**, **Relevance**, **Coherence**,
        **Clarity**, **Logical**, and **Flexibility**.
        """

    for query, answer1, answer2 in tqdm(
        zip(queries, A_answers, B_answers),
        desc="Selection evaluation",
        total=len(queries),
    ):
        prompt = f"""
            You will evaluate two answers to the same question based on eight criteria: **Comprehensiveness**, **Empowerment**, **Accuracy**, **Relevance**, **Coherence**,
            **Clarity**, **Logical**, and **Flexibility**.

            - **Comprehensiveness**: How much detail does the answer provide to cover all aspects of the question?
            - **Empowerment**: How well does the answer help the reader understand and make informed judgments?
            - **Accuracy**: How well does the answer align with factual truth and avoid hallucination?
            - **Relevance**: How precisely does the answer address the core aspects of the question?
            - **Coherence**: How well does the answer integrate and synthesize information into a logically flowing response?
            - **Clarity**: How well does the answer provide complete information while avoiding unnecessary verbosity?
            - **Logical**: How well does the answer maintain consistent logical arguments without contradicting itself?
            - **Flexibility**: How well does the answer handle various question formats, tones, and levels of complexity?

            For each criterion, choose the better answer (either Answer 1 or Answer 2) and explain why.

            Here is the question:
            {query}

            Here are the two answers:

            **Answer 1:**
            {answer1}

            **Answer 2:**
            {answer2}

            Evaluate both answers and provide detailed explanations for each criterion.

            Output your evaluation in the following JSON format:
            {{
                "Comprehensiveness": {{
                    "Winner": "[Answer 1 or Answer 2]",
                    "Explanation": "[Provide explanation here]"
                }},
                "Empowerment": {{
                    "Winner": "[Answer 1 or Answer 2]",
                    "Explanation": "[Provide explanation here]"
                }},
                "Accuracy": {{
                    "Winner": "[Answer 1 or Answer 2]",
                    "Explanation": "[Provide explanation here]"
                }},
                "Relevance": {{
                    "Winner": "[Answer 1 or Answer 2]",
                    "Explanation": "[Provide explanation here]"
                }},
                "Coherence": {{
                    "Winner": "[Answer 1 or Answer 2]",
                    "Explanation": "[Provide explanation here]"
                }},
                "Clarity": {{
                    "Winner": "[Answer 1 or Answer 2]",
                    "Explanation": "[Provide explanation here]"
                }},
                "Logical": {{
                    "Winner": "[Answer 1 or Answer 2]",
                    "Explanation": "[Provide explanation here]"
                }},
                "Flexibility": {{
                    "Winner": "[Answer 1 or Answer 2]",
                    "Explanation": "[Provide explanation here]"
                }}
            }}
        """
        try:
            response = llm_model_func(prompt, sys_prompt)
            responses.append(response)
        except Exception as e:
            print(f"\n  [Error] Evaluation failed: {e}")
            responses.append("{}")

    print(f"\n{len(responses)} responses evaluated.")
    return responses


def fetch_selection_results(responses, A_label, B_label):
    metric_name_list = [
        "Comprehensiveness", "Empowerment", "Accuracy", "Relevance",
        "Coherence", "Clarity", "Logical", "Flexibility", "Averaged Score",
    ]
    total_scores = [0] * 8
    valid_count = 0

    for response in responses:
        try:
            scores = re.findall(r'"Winner":\s*"([^"]+)"', response)
            if len(scores) >= 8:
                for i in range(8):
                    if scores[i].lower() == "answer 1":
                        total_scores[i] += 1
                valid_count += 1
        except Exception:
            continue

    if valid_count == 0:
        print("[Warning] No valid selection results found.")
        return

    total_scores = np.array(total_scores, dtype=float) / valid_count
    total_scores = np.append(total_scores, np.mean(total_scores))

    print(f"\n{'='*50}")
    print(f"Selection Results: {A_label} vs {B_label} ({valid_count}/{len(responses)} valid)")
    print(f"{'='*50}")
    for metric_name, score in zip(metric_name_list, total_scores):
        print(f"  {metric_name:20}: {A_label} {score:.2f} vs. {B_label} {1 - score:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Selection-based evaluation")
    parser.add_argument("-d", "--data_name", type=str, default="mix")
    parser.add_argument("-s", "--stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("-a", "--mode_a", type=str, default="v81",
                        help="Mode label for Answer 1")
    parser.add_argument("-b", "--mode_b", type=str, default="naive",
                        help="Mode label for Answer 2")
    args = parser.parse_args()

    caches_root = Path(CACHES_DIR)
    A_file = caches_root / args.data_name / "response" / f"{args.mode_a}_{args.stage}_stage_result.json"
    B_file = caches_root / args.data_name / "response" / f"{args.mode_b}_{args.stage}_stage_result.json"

    if not A_file.exists():
        print(f"[Error] Answer file A not found: {A_file}")
        sys.exit(1)
    if not B_file.exists():
        print(f"[Error] Answer file B not found: {B_file}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Step 5: Selection evaluation for '{args.data_name}' ({args.stage}-stage)")
    print(f"  Answer 1 ({args.mode_a}): {A_file.name}")
    print(f"  Answer 2 ({args.mode_b}): {B_file.name}")
    print(f"{'='*60}")

    A_queries, A_answers = extract_queries_and_answers(A_file)
    B_queries, B_answers = extract_queries_and_answers(B_file)

    assert len(A_queries) == len(B_queries), f"Query count mismatch: {len(A_queries)} vs {len(B_queries)}"

    # 评估 (A vs B)
    responses = exam_by_selection(A_queries, A_answers, B_answers)

    # 保存
    eval_dir = caches_root / args.data_name / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_file = eval_dir / f"selection_{args.stage}_stage_{args.mode_a}_vs_{args.mode_b}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4)
    print(f"\nResults saved to {output_file}")

    # 计算胜率
    fetch_selection_results(responses, args.mode_a, args.mode_b)
