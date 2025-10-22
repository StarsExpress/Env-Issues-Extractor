from utils import read_articles, unify_issues, save_unified_issues, load_raw_issues, get_distinct_issues
from prompt_engineering import extract_issues, init_gen_pipeline
from k_means import decide_optimal_clusters


def main():
    articles = read_articles()
    gen_pipeline = init_gen_pipeline()
    extract_issues(articles, gen_pipeline)  # streams raw outputs

    distinct_issues = get_distinct_issues()
    decide_optimal_clusters(list(distinct_issues))

    # load raw issues again, unify, and save
    unified = unify_issues(load_raw_issues())
    save_unified_issues(unified)


if __name__ == "__main__":
    main()
