import json
from configs.paths_config import TEACHER_EXTRACTED_ISSUES_PATH, REQUIRED_KEYS
from configs.paths_config import ISSUES2INDICES_PATH, INDICES2ISSUES_PATH


def load_jsonl(path: str) -> list[dict]:
    """Loads JSONL file and returns a list of dictionaries."""
    data = []
    with open(path, "r") as f:
        for line in f:
            data.append(json.loads(line))

    return data


def save_jsonl(path: str, data: list[dict]) -> None:
    """Saves list of dictionaries to JSONL file."""
    with open(path, "w") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_teacher_issues_scores() -> list[dict[str, str | dict[str, int]]]:
    """
    Reads teacher extracted issues from JSONL file and returns a list of dictionaries.
    Each dictionary has the structure:
    {
        "title": str,
        "issues": {
            "issue_name": int,  # severity score
            ...
        }
    }
    This function ensures that each dictionary contains these required keys.
    """
    extracted_issues: list[dict[str, str | dict[str, int]]] = []
    with open(TEACHER_EXTRACTED_ISSUES_PATH) as f:
        for line in f:
            issues_dict = json.loads(line)
            if set(issues_dict.keys()) == REQUIRED_KEYS:
                extracted_issues.append(issues_dict)

    print("Successfully read teacher extracted issues.")
    return extracted_issues


def vectorize_issues_scores(
    extracted_issues: list[dict[str, str | dict[str, int]]]
) -> list[dict[str, list[int]]]:
    """
    Vectorizes the issues and severity scores from the teacher model output.

    Returns:
        encoded: list of dicts, each with:
            - title: str
            - issues: list[int]  (0 or 1 label)
            - severity: list[int]  (0 ~ 10 scores)
    """
    occurred_issues = set()  # Collect all distinct issues.
    for item in extracted_issues:
        occurred_issues.update(set(item["issues"].keys()))

    occurred_issues = sorted(list(occurred_issues))  # Sort list for consistent orders.

    issues2indices = {issue: i for i, issue in enumerate(occurred_issues)}
    json.dump(issues2indices, open(ISSUES2INDICES_PATH, "w"))
    print("Successfully saved issues to indices map.")

    indices2issues = {i: issue for i, issue in enumerate(occurred_issues)}
    json.dump(indices2issues, open(INDICES2ISSUES_PATH, "w"))
    print("Successfully saved indices to issues map.")

    encoded_issues_scores: list[dict[str, list[int]]] = []

    for item in extracted_issues:
        title = item["title"]
        issues_scores = item["issues"]

        issues_vector = [0] * len(occurred_issues)
        severity_vector = [0] * len(occurred_issues)

        for issue, score in issues_scores.items():
            idx = issues2indices[issue]
            issues_vector[idx] = 1  # 0 or 1 label.
            severity_vector[idx] = score  # 1 ~ 10 label.

        encoded_issues_scores.append(
            {
                "title": title,
                "issues": issues_vector,
                "severity": severity_vector
            }
        )

    print("Successfully vectorized issues and scores.")
    print(f"Total unique issues: {len(occurred_issues)}.")
    return encoded_issues_scores


if __name__ == "__main__":
    teacher_issues = read_teacher_issues_scores()
    vectorize_issues_scores(teacher_issues)
