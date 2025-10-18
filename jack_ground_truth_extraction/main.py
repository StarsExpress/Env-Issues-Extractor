from utils import read_articles, unify_issues, save_unified_issues
from prompt_engineering import extract_issues, init_gen_pipeline
from k_means import decide_optimal_clusters


def main():
    articles = read_articles()
    gen_pipeline = init_gen_pipeline()
    issues2scores_list = extract_issues(articles, gen_pipeline)

    distinct_issues = set()
    for issues2scores in issues2scores_list:
        for issue in issues2scores.keys():
            distinct_issues.add(issue)

    decide_optimal_clusters(list(distinct_issues))
    unified_issues2scores_list = unify_issues(issues2scores_list)
    save_unified_issues(unified_issues2scores_list)


if __name__ == "__main__":
    main()
