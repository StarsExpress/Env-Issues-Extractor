import json
from tqdm import tqdm
from datasets import load_dataset
import torch
import numpy as np
from configs.paths_config import STUDENT_EXTRACTED_ISSUES_PATH, TEACHER_EXTRACTED_ISSUES_PATH
from configs.paths_config import SFT_TEST_PATH, EVALUATION_RESULTS_PATH


# -------------------------------------------------------------
# Load teacher ground truth (vectorized test set).
# -------------------------------------------------------------
def load_teacher_extracted_issues(path):
    extracted_issues = []
    with open(path, "r") as f:
        for line in f:
            extracted_issues.append(json.loads(line))

    return extracted_issues


def extract_json(text):
    """
    Extract the JSON dictionary from model output.
    """
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
        return json.loads(text[start:end])

    except:
        return {}


# -------------------------------------------------------------
# Let student model predict issues + severity.
# -------------------------------------------------------------
def run_student_inference(model, tokenizer):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("Running student model inference...")

    dataset = load_dataset("json", data_files={"test": SFT_TEST_PATH})["test"]
    extracted_issues = []

    for row in tqdm(dataset):
        messages = row["messages"]  # Full SFT-style messages.
        user_prompt = messages[0]["content"]  # Only user content.

        input_ids = tokenizer(
            user_prompt,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            out = model.generate(
                **input_ids,
                max_new_tokens=300,
                temperature=0.2
            )

        decoded = tokenizer.decode(out[0], skip_special_tokens=True)

        # Try extract JSON only.
        json_part = extract_json(decoded)

        extracted_issues.append(
            {
                "title": row["title"],
                "issues": json_part
            }
        )

    with open(STUDENT_EXTRACTED_ISSUES_PATH, "w") as f:
        for item in extracted_issues:
            f.write(json.dumps(item) + "\n")

    print(f"Saved student_extracted_issues.jsonl to {STUDENT_EXTRACTED_ISSUES_PATH}")
    return extracted_issues


# -------------------------------------------------------------
# Evaluation for each extracted issue.
# -------------------------------------------------------------
def compute_errors(teacher_extraction, student_extraction):
    """
    teacher_dict = {"air pollution": 7, "deforestation": 6, ...}
    student_dict = {"air pollution": 9, "biodiversity loss": 3}

    Error = sum of |student - teacher| for:
    - overlapping issues
    - extra issues predicted by student  (teacher score = 0)
    - teacher issues missed by student  (student score = 0)
    """
    all_keys = set(teacher_extraction.keys()) | set(student_extraction.keys())

    errors = []
    for k in all_keys:
        teacher = teacher_extraction.get(k, 0)
        student = student_extraction.get(k, 0)
        errors.append(abs(student - teacher))

    return errors


# -------------------------------------------------------------
# Evaluate whole dataset.
# -------------------------------------------------------------
def evaluate_llm():
    teacher_extraction = []
    with open(TEACHER_EXTRACTED_ISSUES_PATH, "r") as f:
        for line in f:
            teacher_extraction.append(json.loads(line))

    student_extraction = []
    with open(STUDENT_EXTRACTED_ISSUES_PATH, "r") as f:
        for line in f:
            student_extraction.append(json.loads(line))

    assert len(teacher_extraction) == len(student_extraction), "Teacher and student test sets mismatch!"

    all_errors = []

    for teacher, student in zip(teacher_extraction, student_extraction):
        teacher_scores = teacher["issues"]
        student_scores = student["issues"]
        errors = compute_errors(teacher_scores, student_scores)
        all_errors.extend(errors)

    all_errors = np.array(all_errors)
    mae = np.mean(all_errors)
    rmse = np.sqrt(np.mean(all_errors ** 2))

    result = {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "Total comparisons": len(all_errors)
    }

    with open(EVALUATION_RESULTS_PATH, "w") as f:
        json.dump(result, f, indent=4)

    print("==================================")
    print("FINAL EVALUATION METRICS:")
    print("==================================")
    print(json.dumps(result, indent=4))
    print(f"Saved evaluation results to {EVALUATION_RESULTS_PATH}")
