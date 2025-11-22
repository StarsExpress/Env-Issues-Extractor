import os

BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_FOLDER = os.path.join(BASE_PATH, "data")
TEACHER_EXTRACTED_ISSUES_PATH = os.path.join(DATA_FOLDER, "teacher_extracted_issues_scores.jsonl")

REQUIRED_KEYS = {"title", "issues"}  # Each line of teacher extracted issues must have exactly these keys.

ISSUES2INDICES_PATH = os.path.join(DATA_FOLDER, "issues2indices.json")
INDICES2ISSUES_PATH = os.path.join(DATA_FOLDER, "indices2issues.json")

VECTORIZED_TRAIN_PATH = os.path.join(DATA_FOLDER, "vectorized_train_data.jsonl")
VECTORIZED_TEST_PATH = os.path.join(DATA_FOLDER, "vectorized_test_data.jsonl")

SFT_TRAIN_PATH = os.path.join(DATA_FOLDER, "sft_train_data.jsonl")
SFT_TEST_PATH = os.path.join(DATA_FOLDER, "sft_test_data.jsonl")


MODEL_FOLDER = os.path.join(BASE_PATH, "model")


OUTPUT_FOLDER = os.path.join(BASE_PATH, "output")
STUDENT_EXTRACTED_ISSUES_PATH = os.path.join(OUTPUT_FOLDER, "student_extracted_issues.jsonl")
EVALUATION_RESULTS_PATH = os.path.join(OUTPUT_FOLDER, "evaluation_results.jsonl")
