import os
import json
import numpy as np
from tqdm import tqdm
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.paths_config import STUDENT_EXTRACTED_ISSUES_PATH, TEACHER_EXTRACTED_ISSUES_PATH
from configs.paths_config import SFT_TEST_PATH, MODEL_FOLDER, EVALUATION_RESULTS_PATH
from utils import load_jsonl


# ============================================================
# Force disable flash attention.
# ============================================================
os.environ["FLASH_ATTENTION"] = "0"
os.environ["ENABLE_FLASH_ATTENTION"] = "0"
os.environ["TORCH_SDP_ATTENTION"] = "0"
os.environ["ATTN_BACKEND"] = "EAGER"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def extract_json(decoded_inference):
    try:
        start = decoded_inference.index("{")
        end = decoded_inference.rindex("}") + 1
        return json.loads(decoded_inference[start:end])

    except Exception:
        return {}


# ============================================================
# Load finetuned model: Qwen2 + LoRA.
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
        attn_implementation="eager"  # Can;t use FlashAttention2.
    )

    return model, tokenizer


def run_student_inference(model, tokenizer):
    print("\nRunning student inference...\n")

    dataset = load_dataset("json", data_files={"test": SFT_TEST_PATH})["test"]
    student_inferences = []

    for row in tqdm(dataset):
        user_prompt = row["messages"][0]["content"]

        inputs = tokenizer(
            user_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(model.device)

        with torch.no_grad():
            with torch.backends.cuda.sdp_kernel(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False
            ):
                out = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=False,
                )

        decoded_inference = tokenizer.decode(out[0], skip_special_tokens=True)
        student_inference = extract_json(decoded_inference)

        student_inferences.append(
            {
                "title": row["title"],
                "issues": student_inference
            }
        )

    with open(STUDENT_EXTRACTED_ISSUES_PATH, "w") as f:
        for x in student_inferences:
            f.write(json.dumps(x) + "\n")

    print(f"\nSaved student extracted issues to {STUDENT_EXTRACTED_ISSUES_PATH}")
    return student_inferences


def compute_errors(teacher_issues, student_issues):
    all_keys = set(teacher_issues.keys()) | set(student_issues.keys())
    errors = []

    for k in all_keys:
        teacher = teacher_issues.get(k, 0)
        student = student_issues.get(k, 0)
        errors.append(abs(teacher - student))

    return errors


def evaluate_llm():
    print("\nEvaluating...\n")

    teacher_issues = load_jsonl(TEACHER_EXTRACTED_ISSUES_PATH)
    student_issues = load_jsonl(STUDENT_EXTRACTED_ISSUES_PATH)

    if len(teacher_issues) != len(student_issues):
        raise ValueError("Teacher and student test size mismatch!")

    all_errors = []
    for teacher, student in zip(teacher_issues, student_issues):
        all_errors.extend(compute_errors(teacher["issues"], student["issues"]))

    errors = np.array(all_errors)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean(errors ** 2)))

    evaluation_metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Total comparisons": len(errors)
    }

    with open(EVALUATION_RESULTS_PATH, "w") as f:
        json.dump(evaluation_metrics, f, indent=4)

    print(json.dumps(evaluation_metrics, indent=4))
    print(f"Saved evaluation results to {EVALUATION_RESULTS_PATH}")
