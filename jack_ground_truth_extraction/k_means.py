from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from kneed import KneeLocator
import joblib
import json
from configs.paths_config import OPTIMAL_PARAMS_PATH, K_MEANS_MODEL_PATH, ISSUES_TO_NAMES_PATH
from configs.k_means_config import MIN_CLUSTERS, RANDOM_STATE


def decide_optimal_clusters(issues_lists: list[str]) -> None:
    model = SentenceTransformer("all-MiniLM-L6-v2")
    issues_vectors = model.encode(issues_lists)

    squared_errors_sum = []
    K_range = range(MIN_CLUSTERS, len(issues_lists) + 1)
    for k in K_range:
        k_means = KMeans(n_clusters=k, random_state=RANDOM_STATE).fit(issues_vectors)
        squared_errors_sum.append(k_means.inertia_)

    knee_locator = KneeLocator(list(K_range), squared_errors_sum, curve="convex", direction="decreasing")
    optimal_clusters = knee_locator.knee

    optimal_k_means = KMeans(n_clusters=optimal_clusters, random_state=RANDOM_STATE)
    issues_labels = optimal_k_means.fit_predict(issues_vectors)

    labels2names = dict()  # Each cluster label maps to a representative issue name.
    issues2names = dict()  # Each issue maps to a representative issue name.

    for issue, label in zip(issues_lists, issues_labels):
        if label not in labels2names.keys():
            labels2names.update({label: issue})
        issues2names.update({issue: labels2names[label]})

    optimal_params = json.loads(OPTIMAL_PARAMS_PATH)
    optimal_params['optimal_clusters'] = optimal_clusters
    with open(OPTIMAL_PARAMS_PATH, "w") as f:
        json.dump(optimal_params, f, indent=4)
    print("Optimal number of clusters saved.")

    joblib.dump(optimal_k_means, K_MEANS_MODEL_PATH)
    print("K Means model at optimal number of clusters saved.")

    with open(ISSUES_TO_NAMES_PATH, "w") as f:
        json.dump(issues2names, f, indent=4)
    print("Issues to names mapper saved.")
