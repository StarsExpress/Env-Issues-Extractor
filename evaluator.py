import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.evaluation_config import EVAL_BATCH_SIZE, PADDING, TRUNCATION, MAX_LEN, MAX_NEW_TOKENS
from configs.paths_config import (
    SFT_TEST_PATH,
    VECTORIZED_TEST_PATH,
    ISSUES2INDICES_PATH,
    MODEL_FOLDER,
    STUDENT_EXTRACTED_ISSUES_PATH,
    EVALUATION_RESULTS_PATH,
)
from utils import load_jsonl


def parse_two_sparse_lists(text: str, vocab_size: int) -> tuple[list[int], list[int]]:
    """
    Expected LLM output text format:
        [23, 120, 501]  # Indices of detected issues.
        [7, 3, 2]  # Corresponding severities ranging from 1~10.

    Returns two lists of length `vocab_size`:
        - issues_vector: binary vector indicating presence of issues.
                        1 in issues_vector indicates presence of the issue.
                        0 in issues_vector indicates absence of the issue.

        - severity_vector: integer vector indicating severity levels.
                           0 in severity_vector indicates absence of the issue.
                          1~10 in severity_vector indicates severity level.
    """
    text_flat = text.replace("\n", " ")
    matches = re.findall(r"\[.*?\]", text_flat)

    if len(matches) < 2:
        return [0] * vocab_size, [0] * vocab_size

    try:
        indices = json.loads(matches[0])
        severities = json.loads(matches[1])

        if not isinstance(indices, list) or not isinstance(severities, list):  # Failed to parse.
            return [0] * vocab_size, [0] * vocab_size

        if len(indices) != len(severities):  # Length mismatch.
            return [0] * vocab_size, [0] * vocab_size

        issues_vector = [0] * vocab_size
        severity_vector = [0] * vocab_size

        for idx, severity in zip(indices, severities):
            if 0 <= idx < vocab_size and 1 <= severity <= 10:  # Valid index and severity.
                issues_vector[idx] = 1
                severity_vector[idx] = severity

        return issues_vector, severity_vector

    except:  # Any parsing error.
        return [0] * vocab_size, [0] * vocab_size


def load_fine_tuned_model() -> tuple:
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_FOLDER,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_FOLDER,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    return model, tokenizer


def run_inference(model, tokenizer, vocabularies: list[str]) -> list[dict]:
    vocab_size = len(vocabularies)
    test_prompts = load_jsonl(SFT_TEST_PATH)

    inference_prompts = []
    titles = []

    for item in test_prompts:
        messages = item["messages"]
        title = item["title"]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        titles.append(title)
        inference_prompts.append(prompt)

    # ---------- Batch inference. ----------
    student_vectors = []
    student_extracted_issues = []

    print("\nRunning batch student inference...\n")

    for idx in tqdm(range(0, len(inference_prompts), EVAL_BATCH_SIZE)):
        batch_prompts = inference_prompts[idx : idx + EVAL_BATCH_SIZE]
        batch_titles = titles[idx : idx + EVAL_BATCH_SIZE]

        # Tokenize as batch.
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=PADDING,
            truncation=TRUNCATION,
            max_length=MAX_LEN,
        ).to(model.device)

        with torch.no_grad():
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    use_cache=True
                )

        input_ids = batch_inputs["input_ids"]  # Slice out generation part.
        for batch_num in range(len(batch_prompts)):
            gen_tokens = outputs[batch_num][ input_ids[batch_num].shape[0] : ]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Parse sparse lists.
            issues_vector, severity_vector = parse_two_sparse_lists(decoded, vocab_size)

            # Build extracted issues dict.
            extracted_dict = {
                vocabularies[idx]: severity
                for idx, severity in enumerate(severity_vector) if severity > 0
            }

            student_extracted_issues.append({
                "title": batch_titles[batch_num],
                "issues": extracted_dict
            })

            student_vectors.append({
                "issues": issues_vector,
                "severity": severity_vector
            })

    with open(STUDENT_EXTRACTED_ISSUES_PATH, "w") as f:
        for row in student_extracted_issues:
            f.write(json.dumps(row) + "\n")

    print(f"\nSaved student extractions to {STUDENT_EXTRACTED_ISSUES_PATH}")
    return student_vectors


# ============================================================
# Compute MAE / RMSE.
# ============================================================
def compute_metrics(teacher_vectors: list[dict], student_vectors: list[dict]) -> tuple[float, float]:
    mae_sum = 0
    mse_sum = 0
    count = 0

    for teacher_vec, student_vec in zip(teacher_vectors, student_vectors):
        teacher_severity = teacher_vec["severity"]
        student_severity = student_vec["severity"]

        diffs = [(s - t) for s, t in zip(student_severity, teacher_severity)]

        mae_sum += sum(abs(x) for x in diffs) / len(teacher_severity)
        mse_sum += sum(x * x for x in diffs) / len(teacher_severity)
        count += 1

    return mae_sum / count, mse_sum / count


def evaluate_llm():
    print("Loading model...")
    model, tokenizer = load_fine_tuned_model()

    print("Loading vocab...")
    issues2indices = json.load(open(ISSUES2INDICES_PATH))
    vocabularies = list(issues2indices.keys())

    print("Loading teacher vectors...")
    teacher_vectors = load_jsonl(VECTORIZED_TEST_PATH)

    print("Running student inference (batch)...")
    student_vectors = run_inference(model, tokenizer, vocabularies)

    print("Computing MAE / RMSE...")
    mae, rmse = compute_metrics(teacher_vectors, student_vectors)

    results = {
        "MAE": mae,
        "RMSE": rmse,
        "Total samples": len(teacher_vectors)
    }

    with open(EVALUATION_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("\n===== Evaluation Results =====")
    print(json.dumps(results, indent=4))
    print(f"\nSaved evaluation to {EVALUATION_RESULTS_PATH}")
