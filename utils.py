import json
import pandas as pd
from configs.paths_config import ISSUES_TO_NAMES_PATH, NEWS_CSV_PATH, ISSUES_SCORES_CSV_PATH


def read_articles() -> list[str]:
    news_df = pd.read_csv(NEWS_CSV_PATH)["Article Text"].dropna()
    print("Successfully read articles and drop NaN.")
    return news_df.to_list()

def save_extracted_issues(articles_issues2scores_list: list[tuple[str, dict[str, int]]]) -> None:
    records = [
        {"Article Text": article, "Issues Scores": json.dumps(issues2scores, ensure_ascii=False)}
        for article, issues2scores in articles_issues2scores_list
    ]
    extracted_issues_df = pd.DataFrame.from_records(records)
    extracted_issues_df.to_csv(ISSUES_SCORES_CSV_PATH, index=False)
    print("Successfully saved extracted issues scores.")

def unify_issues(issues2scores_list: list[dict[str, int]]) -> list[dict[str, int]]:
    issues2names = json.loads(ISSUES_TO_NAMES_PATH)
    unified_issues2scores_list = []

    for issues2scores in issues2scores_list:
        renamed_issues2scores = dict()
        for issue, score in issues2scores.items():
            if issue in issues2names:
                issue = issues2names[issue]
            renamed_issues2scores[issue] = score

        unified_issues2scores_list.append(renamed_issues2scores)

    print("Successfully unified issues.")
    return unified_issues2scores_list

def save_unified_issues(unified_issues2scores_list: list[dict[str, int]]) -> None:
    news_df = pd.read_csv(NEWS_CSV_PATH)["Article Text"].dropna()
    news_df["Issues Scores"] = unified_issues2scores_list
    news_df.to_csv(ISSUES_SCORES_CSV_PATH, index=False)
    print("Successfully saved new issues scores.")
