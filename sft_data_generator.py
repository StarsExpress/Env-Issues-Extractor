import json
from configs.paths_config import (
    VECTORIZED_TRAIN_PATH,
    VECTORIZED_TEST_PATH,
    ISSUES2INDICES_PATH,
    SFT_TRAIN_PATH,
    SFT_TEST_PATH
)
from utils import load_jsonl, save_jsonl


def build_sft_dataset(vectorized_data, vocabularies: list[str]):
    sft_dataset = []

    for sample in vectorized_data:
        title = str(sample["title"]).strip()
        issues_vec = sample["issues"]
        severity_vec = sample["severity"]

        # ---- Sparse encoding. ----
        indices = [i for i, v in enumerate(issues_vec) if v == 1]
        severities = [severity_vec[i] for i in indices]

        # Convert to shorter JSON lists.
        assistant_output = json.dumps(indices) + "\n" + json.dumps(severities)

        # ---- Build prompt. ----
        vocab_len = len(vocabularies)
        user_prompt = (
            "You are an environmental news analyst.\n"
            "Your task is to output TWO lists ONLY:\n\n"
            "Line 1: [indices of issues present]\n"
            "Line 2: [severity scores aligned with the same indices]\n\n"
            f"Valid issue indices: 0 to {vocab_len - 1}\n"
            "Severity scores range: 1–10\n\n"
            f"### Article title:\n{title}\n\n"
            "### Output:\n"
            "(Two JSON lists ONLY, no explanation or text)"
        )

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


def save_sft_data():
    vector_train = load_jsonl(VECTORIZED_TRAIN_PATH)
    vector_test = load_jsonl(VECTORIZED_TEST_PATH)

    issues2indices = json.load(open(ISSUES2INDICES_PATH))
    issues = list(issues2indices.keys())  # Keys already sorted during utils vectorization.

    sft_train = build_sft_dataset(vector_train, issues)
    sft_test = build_sft_dataset(vector_test, issues)

    save_jsonl(SFT_TRAIN_PATH, sft_train)
    save_jsonl(SFT_TEST_PATH, sft_test)

    print(f"SFT train dataset saved at {SFT_TRAIN_PATH} with {len(sft_train)} samples.")
    print(f"SFT test dataset saved at {SFT_TEST_PATH} with {len(sft_test)} samples.")


if __name__ == "__main__":
    save_sft_data()
