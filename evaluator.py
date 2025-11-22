import json
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.evaluation_config import BATCH_SIZE
from configs.paths_config import (
    SFT_TEST_PATH,
    VECTORIZED_TEST_PATH,
    ISSUES2INDICES_PATH,
    MODEL_FOLDER,
    STUDENT_EXTRACTED_ISSUES_PATH,
    EVALUATION_RESULTS_PATH,
)
from utils import load_jsonl


# ============================================================
# Sparse parser.
# ============================================================
def parse_two_sparse_lists(text, vocab_size):
    """
    Expected model output format:
        [23, 120, 501]
        [7, 3, 2]
    """
    text_flat = text.replace("\n", " ")
    matches = re.findall(r"\[.*?\]", text_flat)

    if len(matches) < 2:
        return [0] * vocab_size, [0] * vocab_size

    try:
        indices = json.loads(matches[0])
        severities = json.loads(matches[1])

        if not isinstance(indices, list) or not isinstance(severities, list):
            return [0] * vocab_size, [0] * vocab_size

        if len(indices) != len(severities):
            return [0] * vocab_size, [0] * vocab_size

        issues_vec = [0] * vocab_size
        severity_vec = [0] * vocab_size

        for idx, sev in zip(indices, severities):
            if 0 <= idx < vocab_size and 1 <= sev <= 10:
                issues_vec[idx] = 1
                severity_vec[idx] = sev

        return issues_vec, severity_vec

    except:
        return [0] * vocab_size, [0] * vocab_size


# ============================================================
# Load fine-tuned student model.
# ============================================================
def load_student_model():
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


def run_student_inference(model, tokenizer, vocabularies):
    vocab_size = len(vocabularies)
    test_prompts = load_jsonl(SFT_TEST_PATH)

    prompts = []
    titles = []
    # ---------- Build prompts first. ----------
    for item in test_prompts:
        messages = item["messages"]
        title = item["title"]

        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        titles.append(title)
        prompts.append(prompt)

    # ---------- Batch inference. ----------
    student_vectors = []
    student_extracted_issues = []

    print("\nRunning batch student inference...\n")

    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        batch_prompts = prompts[i : i + BATCH_SIZE]
        batch_titles = titles[i : i + BATCH_SIZE]

        # Tokenize as batch.
        batch_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False, enable_math=True, enable_mem_efficient=False
            ):
                outputs = model.generate(
                    **batch_inputs,
                    max_new_tokens=64,
                    do_sample=False,
                    use_cache=True
                )

        input_ids = batch_inputs["input_ids"]  # Slice out generation part.
        for b in range(len(batch_prompts)):
            gen_tokens = outputs[b][ input_ids[b].shape[0] : ]
            decoded = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            # Parse sparse lists.
            issues_vec, severity_vec = parse_two_sparse_lists(decoded, vocab_size)

            # Build dict.
            extracted_dict = {
                vocabularies[idx]: sev
                for idx, sev in enumerate(severity_vec) if sev > 0
            }

            student_extracted_issues.append({
                "title": batch_titles[b],
                "issues": extracted_dict
            })

            student_vectors.append({
                "issues": issues_vec,
                "severity": severity_vec
            })

    with open(STUDENT_EXTRACTED_ISSUES_PATH, "w") as f:
        for row in student_extracted_issues:
            f.write(json.dumps(row) + "\n")

    print(f"\nSaved student extractions to {STUDENT_EXTRACTED_ISSUES_PATH}")
    return student_vectors


# ============================================================
# Compute MAE / RMSE.
# ============================================================
def compute_metrics(teacher_vecs, student_vecs):
    mae_sum = 0
    mse_sum = 0
    count = 0

    for teacher_vec, student_vec in zip(teacher_vecs, student_vecs):
        teacher_sev = teacher_vec["severity"]
        student_sev = student_vec["severity"]

        diffs = [(s - t) for s, t in zip(student_sev, teacher_sev)]

        mae_sum += sum(abs(x) for x in diffs) / len(teacher_sev)
        mse_sum += sum(x * x for x in diffs) / len(teacher_sev)
        count += 1

    return mae_sum / count, mse_sum / count


def evaluate_llm():
    print("Loading model...")
    model, tokenizer = load_student_model()

    print("Loading vocab...")
    issues2indices = json.load(open(ISSUES2INDICES_PATH))
    vocabularies = list(issues2indices.keys())

    print("Loading teacher vectors...")
    teacher_vecs = load_jsonl(VECTORIZED_TEST_PATH)

    print("Running student inference (batch)...")
    student_vecs = run_student_inference(model, tokenizer, vocabularies)

    print("Computing MAE / RMSE...")
    mae, rmse = compute_metrics(teacher_vecs, student_vecs)

    results = {
        "MAE": mae,
        "RMSE": rmse,
        "Total samples": len(teacher_vecs)
    }

    with open(EVALUATION_RESULTS_PATH, "w") as f:
        json.dump(results, f, indent=4)

    print("\n===== Evaluation Results =====")
    print(json.dumps(results, indent=4))
    print(f"\nSaved evaluation to {EVALUATION_RESULTS_PATH}")
