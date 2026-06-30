"""
Step 2: 问题生成
使用 GPT-4o-mini 从原始语料中自动生成 1/2/3 阶段的测试问题。
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
from my_config import EVAL_LLM_API_KEY, EVAL_LLM_BASE_URL, EVAL_LLM_MODEL, CACHES_DIR, MAX_QUESTIONS, LEN_BIG_CHUNKS


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


# ============================================================================
# 三种难度的问题 Prompt
# ============================================================================

question_prompt = {
    # 单阶段问题
    1: """
            You are a professional teacher, and you are now asked to design a question that meets the requirements based on the reference.
            ################
            Reference:
            Given the following fragment of a data set:
            {context}
            ################
            Requirements:
            1. This question should be of the question-and-answer (QA) type, and no answer is required.
            2. This question mainly tests the details of the information and knowledge in the reference. Avoid general and macro question.
            3. The question must not include any conjunctions such as "specifically", "particularly", "and", "or", "and how", "and what" or similar phrases that imply additional inquiries.
            4. The question must focus on a single aspect or detail from the reference, avoiding the combination of multiple inquiries.
            5. Please design question from the professional perspective and domain factors covered by the reference.
            6. This question need to be meaningful and difficult, avoiding overly simplistic inquiries.
            7. This question should be based on the complete context, so that the respondent knows what you are asking and doesn't get confused.
            8. State the question directly in a single sentence, without statements like "How in this reference?" or "What about this data set?" or "as described in the reference."
            ################
            Output the content of question in the following structure:
            {{
                "Question": [question description],
            }}
        """,
    # 双阶段问题
    2: """
            You are a professional teacher, and your task is to design a single question that contains two interconnected sub-questions, 
            demonstrating a progressive relationship based on the reference.
            ################
            Reference:
            Given the following fragment of a data set:
            {context}
            ################
            Requirements:
            1. This question should be of the question-and-answer (QA) type, and no answer is required.
            2. The question must include two sub-questions connected by transitional phrases such as "and" or "specifically," indicating progression.
            3. Focus on testing the details of the information and knowledge in the reference. Avoid general and macro questions.
            4. Design the question from a professional perspective, considering the domain factors covered by the reference.
            5. Ensure the question is meaningful and challenging, avoiding trivial inquiries.
            6. The question should be based on the complete context, ensuring clarity for the respondent.
            7. State the question directly in a single sentence, without introductory phrases like "How in this reference?" or "What about this data set?".
            ################
            Output the content of the question in the following structure:
            {{
            "Question": [question description],
            }}
        """,
    # 三阶段问题
    3: """
            You are a professional teacher, and your task is to design a single question that contains three interconnected sub-questions, 
            demonstrating a progressive relationship based on the reference.
            ################
            Reference:
            Given the following fragment of a data set:
            {context}
            ################
            Requirements:
            1. This question should be of the question-and-answer (QA) type, and no answer is required.
            2. The question must include three sub-questions connected by transitional phrases such as "and" or "specifically," indicating progression.
            3. Focus on testing the details of the information and knowledge in the reference. Avoid general and macro questions.
            4. Design the question from a professional perspective, considering the domain factors covered by the reference.
            5. Ensure the question is meaningful and challenging, avoiding trivial inquiries.
            6. The question should be based on the complete context, ensuring clarity for the respondent.
            7. State the question directly in a single sentence, without introductory phrases like "How in this reference?" or "What about this data set?".
            ################
            Output the content of the question in the following structure:
            {{
            "Question": [question description],
            }}
        """,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Extract questions from contexts")
    parser.add_argument("-d", "--data_name", type=str, default="mix")
    parser.add_argument("-s", "--stage", type=int, default=2, choices=[1, 2, 3],
                        help="Question stage (1=single, 2=two-stage, 3=three-stage)")
    parser.add_argument("-n", "--num_questions", type=int, default=None,
                        help=f"Number of questions to generate (default: {MAX_QUESTIONS} from config)")
    args = parser.parse_args()

    data_name = args.data_name
    question_stage = args.stage
    max_cnt = args.num_questions if args.num_questions else MAX_QUESTIONS
    len_big_chunks = LEN_BIG_CHUNKS

    caches_root = Path(CACHES_DIR)
    context_file = caches_root / data_name / "contexts" / f"{data_name}_unique_contexts.json"

    if not context_file.exists():
        print(f"[Error] Context file not found: {context_file}")
        print("  请先运行 step_0_preprocess.py")
        sys.exit(1)

    with open(context_file, "r", encoding="utf-8") as f:
        unique_contexts = json.load(f)

    print(f"{'='*60}")
    print(f"Step 2: Generating {max_cnt} {question_stage}-stage questions")
    print(f"  Dataset: {data_name} ({len(unique_contexts)} unique contexts)")
    print(f"  Model:   {EVAL_LLM_MODEL}")
    print(f"{'='*60}")

    question_list, reference_list = [], []
    cnt = 0
    max_idx = max(len(unique_contexts) - len_big_chunks - 1, 1)

    with tqdm(total=max_cnt, desc=f"Extracting {question_stage}-stage questions") as pbar:
        while cnt < max_cnt:
            # 随机选取连续的 context 片段
            idx = np.random.randint(0, max_idx)
            big_chunks = unique_contexts[idx: idx + len_big_chunks]
            context = "".join(big_chunks)

            prompt = question_prompt[question_stage].format(context=context)

            try:
                response = llm_model_func(prompt)
            except Exception as e:
                print(f"\n  [Error] LLM call failed: {e}")
                continue

            question = re.findall(r'"Question":\s*"(.*?)"', response, re.DOTALL)
            if len(question) == 0:
                # 尝试更宽松的匹配
                question = re.findall(r'"Question":\s*["\'](.+?)["\']', response, re.DOTALL)
            if len(question) == 0:
                print(f"\n  [Warning] No question found in response, retrying...")
                continue

            question_list.append(question[0].strip())
            reference_list.append(context)

            cnt += 1
            pbar.update(1)

    # 保存问题和参考答案
    output_dir = caches_root / data_name / "questions"
    output_dir.mkdir(parents=True, exist_ok=True)

    question_file = output_dir / f"{question_stage}_stage.json"
    ref_file = output_dir / f"{question_stage}_stage_ref.json"

    with open(question_file, "w", encoding="utf-8") as f:
        json.dump(question_list, f, ensure_ascii=False, indent=4)
    with open(ref_file, "w", encoding="utf-8") as f:
        json.dump(reference_list, f, ensure_ascii=False, indent=4)

    print(f"\n[Done] {len(question_list)} questions saved to:")
    print(f"  Questions:  {question_file}")
    print(f"  References: {ref_file}")
