import json
import pandas as pd
from configs.paths_config import ISSUES_TO_NAMES_PATH, NEWS_CSV_PATH, ISSUES_SCORES_CSV_PATH, RAW_ISSUES_JSONL_PATH


def read_articles() -> list[tuple[str, str]]:
    news_df = pd.read_csv(NEWS_CSV_PATH).dropna(subset=["Article Text"])  # ensure text present
    print("Successfully read articles and drop NaN.")
    return list(zip(news_df["Title"], news_df["Article Text"]))

def unify_issues(raw_article_dicts: list[dict]):
    issues2names = json.loads(ISSUES_TO_NAMES_PATH)
    unified = []
    for article_dict in raw_article_dicts:
        title = article_dict["title"]
        issues_dict = article_dict["issues"]

        renamed = {}
        for issue, score in issues_dict.items():
            if issue in issues2names:
                issue = issues2names[issue]
            renamed[issue] = score

        unified.append({"title": title, "issues": renamed})

    print("Successfully unified issues.")
    return unified

def save_unified_issues(unified_issues2scores_list: list[dict[str, int]]) -> None:
    news_df = pd.read_csv(NEWS_CSV_PATH)["Article Text"].dropna()
    news_df["Issues Scores"] = unified_issues2scores_list
    news_df.to_csv(ISSUES_SCORES_CSV_PATH, index=False)
    print("Successfully saved new issues scores.")


def append_raw_issues(batch_dicts: list[dict[str, int]]) -> None:
    import json, os
    os.makedirs(os.path.dirname(RAW_ISSUES_JSONL_PATH), exist_ok=True)
    with open(RAW_ISSUES_JSONL_PATH, "a") as f:
        for d in batch_dicts:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")


def load_raw_issues() -> list[dict[str, int]]:
    import json, os
    if not os.path.exists(RAW_ISSUES_JSONL_PATH):
        return []
    with open(RAW_ISSUES_JSONL_PATH) as f:
        return [json.loads(line) for line in f if line.strip()]


def get_distinct_issues() -> set[str]:
    distinct: set[str] = set()
    for d in load_raw_issues():
        for k in d.get("issues", {}).keys():
            distinct.add(k)
    return distinct
