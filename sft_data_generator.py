import json
from configs.paths_config import VECTORIZED_TRAIN_PATH, VECTORIZED_TEST_PATH, ISSUES2INDICES_PATH
from configs.paths_config import SFT_TRAIN_PATH, SFT_TEST_PATH
from utils import load_jsonl, save_jsonl


def build_prompt(title, vocab_list):
    vocab_text = ", ".join(vocab_list)
    return (
        "You are an environmental news analyst.\n"
        "Your task is to output two lists:\n"
        "1. issues: binary list (1 if issue is present, else 0)\n"
        "2. severity: list of scores (0–10)\n\n"
        f"Environmental issue list (in fixed order): [{vocab_text}]\n\n"
        f"### Article title:\n{title}\n"
    )


def build_sft_dataset(vectorized_data, vocabularies: list[str]):
    sft_dataset = []
    for sample in vectorized_data:
        title = sample["title"]

        title = str(title).strip()  # Title must be string.

        issues = sample["issues"]
        severity = sample["severity"]

        user_prompt = build_prompt(title, vocabularies)
        assistant_output = {
            "issues": issues,
            "severity": severity
        }

        sft_dataset.append(
            {
                "title": title,
                "messages": [
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": json.dumps(assistant_output)}
                ]
            }
        )

    return sft_dataset


def save_sft_data():
    vector_train = load_jsonl(VECTORIZED_TRAIN_PATH)
    vector_test = load_jsonl(VECTORIZED_TEST_PATH)

    issues2indices = json.load(open(ISSUES2INDICES_PATH))
    issues = list(issues2indices.keys())

    sft_train = build_sft_dataset(vector_train, issues)
    sft_test = build_sft_dataset(vector_test, issues)

    save_jsonl(SFT_TRAIN_PATH, sft_train)
    save_jsonl(SFT_TEST_PATH, sft_test)

    print(f"SFT train dataset saved at {SFT_TRAIN_PATH} with {len(sft_train)} samples.")
    print(f"SFT test dataset saved at {SFT_TEST_PATH} with {len(sft_test)} samples.")


if __name__ == "__main__":
    save_sft_data()
