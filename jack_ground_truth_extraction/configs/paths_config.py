import os

BASE_PATH = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_FOLDER = os.path.join(BASE_PATH, "data")
NEWS_CSV_PATH = os.path.join(DATA_FOLDER, "guardian_environment_news.csv")

OUTPUT_FOLDER = os.path.join(BASE_PATH, "output")
OPTIMAL_PARAMS_PATH = os.path.join(OUTPUT_FOLDER, "optimal_params.json")
K_MEANS_MODEL_PATH = os.path.join(OUTPUT_FOLDER, "optimal_k_means.pkl")
ISSUES_TO_NAMES_PATH = os.path.join(OUTPUT_FOLDER, "issues_to_names.json")
ISSUES_SCORES_CSV_PATH = os.path.join(OUTPUT_FOLDER, "issues_scores.csv")
