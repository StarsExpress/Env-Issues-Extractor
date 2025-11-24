import json
from configs.paths_config import VECTORIZED_TRAIN_PATH, VECTORIZED_TEST_PATH
from configs.paths_config import ISSUES2INDICES_PATH, SFT_TRAIN_PATH, SFT_TEST_PATH
from utils import load_jsonl, save_jsonl


def build_sft_dataset(vectorized_data: list[dict], vocabularies: list[str], for_train: bool) -> list[dict]:
    """
    Builds supervised fine-tuning dataset from vectorized data.

    Args:
        vectorized_data (list[dict]): List of samples with keys like "title", "issues", and "severity".
        vocabularies (list[str]): List of issue vocabularies for index reference.
        for_train (bool): Flag indicating if the dataset is for training or evaluation.

    Returns:
        list[dict]: SFT dataset formatted for fine-tuning and evaluation.
    """
    sft_dataset = []

    for sample in vectorized_data:
        title = str(sample["title"]).strip()
        issues_vector = sample["issues"]
        severity_vector = sample["severity"]

        if for_train:  # In training, provide assistant output as reference.
            # ---- Sparse encoding. ----
            indices = [idx for idx, entry in enumerate(issues_vector) if entry == 1]
            severities = [severity_vector[idx] for idx in indices]
            assistant_output = json.dumps(indices) + "\n" + json.dumps(severities)  # Convert to shorter JSON lists.

        else:  # In evaluation, don't give any reference.
            assistant_output = ""

        # ---- Build prompt. ----
        vocab_size = len(vocabularies)
        user_prompt = (
            "You are an environmental news analyst.\n"
            "Your task is to output TWO lists ONLY:\n\n"
            "Line 1: [indices of issues present]\n"
            "Line 2: [severity scores aligned with the same indices]\n\n"
            f"Valid issue indices: 0 to {vocab_size - 1}\n"
            "Severity scores range: 1–10\n\n"
            f"### Article title:\n{title}\n\n"
            "### Output:\n"
            "(Two JSON lists ONLY, no explanation or text)"
        )

        # Required prompts format for Qwen.
        sft_dataset.append(
            {
                "title": title,
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_output}
                ]
            }
        )

    return sft_dataset


def save_sft_data() -> None:
    """Loads vectorized data, builds SFT datasets, and saves them."""
    vector_train = load_jsonl(VECTORIZED_TRAIN_PATH)
    vector_test = load_jsonl(VECTORIZED_TEST_PATH)

    issues2indices = json.load(open(ISSUES2INDICES_PATH))
    issues = list(issues2indices.keys())  # Keys already sorted during utils vectorization.

    sft_train = build_sft_dataset(vector_train, issues, for_train=True)
    sft_test = build_sft_dataset(vector_test, issues, for_train=False)

    save_jsonl(SFT_TRAIN_PATH, sft_train)
    save_jsonl(SFT_TEST_PATH, sft_test)

    print(f"SFT train dataset saved at {SFT_TRAIN_PATH} with {len(sft_train)} samples.")
    print(f"SFT test dataset saved at {SFT_TEST_PATH} with {len(sft_test)} samples.")


if __name__ == "__main__":
    from utils import read_teacher_issues_scores, vectorize_issues_scores
    from train_test_preparer import split_train_test

    teacher_issues = read_teacher_issues_scores()
    encoded_issues_scores = vectorize_issues_scores(teacher_issues)
    split_train_test(encoded_issues_scores)
    save_sft_data()
