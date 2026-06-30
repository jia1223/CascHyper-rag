"""
Step 4: Scoring 评估
使用 GPT-4o-mini 从 5 个维度对每个回答打分 (0-100)。
需要参考答案。
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


def extract_queries_and_refs(question_file_path):
    ref_file_path = question_file_path.with_stem(f"{question_file_path.stem}_ref")
    with open(question_file_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(ref_file_path, "r", encoding="utf-8") as f:
        refs = json.load(f)
    return queries, refs


def exam_by_scoring(queries, answers, refs):
    responses = []
    sys_prompt = """
        ---Role---
        You are an expert tasked with evaluating answers to the questions by using the relevant documents based on five criteria:**Comprehensiveness**, **Diversity**,**Empowerment**, **Logical**,and **Readability** .
        """

    for query, answer, reference in tqdm(
        zip(queries, answers, refs), desc="Scoring evaluation", total=len(queries)
    ):
        prompt = f"""
            You will evaluate the answers to the questions by using the relevant documents based on five criteria:**Comprehensiveness**, **Diversity**,**Empowerment**, **Logical**,and **Readability** .

            - **Comprehensiveness** -
            Measure whether the answer comprehensively covers all key aspects of the question and whether there are omissions.
            Level   | score range | description
            Level 1 | 0-20   | The answer is extremely one-sided, leaving out key parts or important aspects of the question.
            Level 2 | 20-40  | The answer has some content, but it misses many important aspects of the question and is not comprehensive enough.
            Level 3 | 40-60  | The answer is more comprehensive, covering the main aspects of the question, but there are still some omissions.
            Level 4 | 60-80  | The answer is comprehensive, covering most aspects of the question, with few omissions.
            Level 5 | 80-100 | The answer is extremely comprehensive, covering all aspects of the question with no omissions.

            - **Diversity** -
            Measure the richness of the answer content, including background knowledge, extended information, case studies, etc.
            Level   | score range | description
            Level 1 | 0-20   | The answer is extremely sparse, providing only direct answers without additional information.
            Level 2 | 20-40  | The answer provides a direct answer but contains only a small amount of relevant knowledge expansion.
            Level 3 | 40-60  | In addition to the direct answers, the answer also provides some relevant background knowledge.
            Level 4 | 60-80  | The answer is rich, providing more relevant background knowledge and supplementary information.
            Level 5 | 80-100 | The answer provides a lot of relevant knowledge, expanded content and in-depth analysis.

            - **Empowerment** -
            Measure the credibility of the answer and whether it convinces the reader that it is correct.
            Level   | score range | description
            Level 1 | 0-20   | The answer lacks credibility, contains obvious errors or false information.
            Level 2 | 20-40  | The answer has some credibility, but some information is not accurate.
            Level 3 | 40-60  | The answer is credible and provides some supporting information.
            Level 4 | 60-80  | The answer is highly credible, providing sufficient supporting information.
            Level 5 | 80-100 | The answer is highly credible with authoritative supporting information.

            - **Logical** -
            Measure whether the answers are coherent, clear, and easy to understand.
            Level   | score range | description
            Level 1 | 0-20   | The answer is illogical, incoherent, and difficult to understand.
            Level 2 | 20-40  | The answer has some logic, but is incoherent in parts.
            Level 3 | 40-60  | The answer is logically clear and basically coherent.
            Level 4 | 60-80  | The answer is logical, coherent, and easy to understand.
            Level 5 | 80-100 | The answer is extremely logical, fluent and well-organized.

            - **Readability** -
            Measure whether the answer is well organized, clear in format, and easy to read.
            Level   | score range | description
            Level 1 | 0-20   | The format is confused and difficult to read.
            Level 2 | 20-40  | There are some problems in the format.
            Level 3 | 40-60  | The format is basically clear.
            Level 4 | 60-80  | The format is clear and well organized.
            Level 5 | 80-100 | The format is very clear with excellent reading experience.

            For each indicator, give a Level and a Score within the level's score range.

            Here are the relevant documents:
                {reference}

            Here are the questions:
                {query}

            Here are the answers:
                {answer}

            Output your evaluation in the following JSON format:
            {{
                "Comprehensiveness": {{
                    "Explanation": "Provide explanation here",
                    "Level": "A single number 1-5",
                    "Score": "A single number 0-100"
                }},
                "Diversity": {{
                    "Explanation": "Provide explanation here",
                    "Level": "A single number 1-5",
                    "Score": "A single number 0-100"
                }},
                "Empowerment": {{
                    "Explanation": "Provide explanation here",
                    "Level": "A single number 1-5",
                    "Score": "A single number 0-100"
                }},
                "Logical": {{
                    "Explanation": "Provide explanation here",
                    "Level": "A single number 1-5",
                    "Score": "A single number 0-100"
                }},
                "Readability": {{
                    "Explanation": "Provide explanation here",
                    "Level": "A single number 1-5",
                    "Score": "A single number 0-100"
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


def fetch_scoring_results(responses):
    metric_name_list = [
        "Comprehensiveness", "Diversity", "Empowerment",
        "Logical", "Readability", "Averaged Score",
    ]
    total_scores = [0.0] * 5
    valid_count = 0

    for response in responses:
        try:
            scores = re.findall(r'"Score":\s*"?(\d+)"?', response)
            if len(scores) >= 5:
                for i in range(5):
                    total_scores[i] += float(scores[i])
                valid_count += 1
        except Exception:
            continue

    if valid_count == 0:
        print("[Warning] No valid scoring results found.")
        return

    total_scores = np.array(total_scores) / valid_count
    total_scores = np.append(total_scores, np.mean(total_scores))

    print(f"\n{'='*40}")
    print(f"Scoring Results ({valid_count}/{len(responses)} valid)")
    print(f"{'='*40}")
    for metric_name, score in zip(metric_name_list, total_scores):
        print(f"  {metric_name:20}: {score:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Scoring-based evaluation")
    parser.add_argument("-d", "--data_name", type=str, default="mix")
    parser.add_argument("-s", "--stage", type=int, default=2, choices=[1, 2, 3])
    parser.add_argument("-m", "--mode", type=str, default="v81",
                        help="Response mode label (e.g. 'v81')")
    args = parser.parse_args()

    caches_root = Path(CACHES_DIR)
    question_file = caches_root / args.data_name / "questions" / f"{args.stage}_stage.json"
    answer_file = caches_root / args.data_name / "response" / f"{args.mode}_{args.stage}_stage_result.json"

    if not question_file.exists():
        print(f"[Error] Question file not found: {question_file}")
        sys.exit(1)
    if not answer_file.exists():
        print(f"[Error] Answer file not found: {answer_file}")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"Step 4: Scoring evaluation for '{args.data_name}' ({args.stage}-stage, mode={args.mode})")
    print(f"{'='*60}")

    raw_queries, raw_refs = extract_queries_and_refs(question_file)
    queries, answers = extract_queries_and_answers(answer_file)

    assert len(queries) == len(raw_queries), f"Query count mismatch: {len(queries)} vs {len(raw_queries)}"
    assert len(queries) == len(raw_refs), f"Ref count mismatch"

    # 评估
    responses = exam_by_scoring(raw_queries, answers, raw_refs)

    # 保存
    eval_dir = caches_root / args.data_name / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    output_file = eval_dir / f"scoring_{args.mode}_{args.stage}_stage.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(responses, f, indent=4)
    print(f"\nResults saved to {output_file}")

    # 计算分数
    fetch_scoring_results(responses)
