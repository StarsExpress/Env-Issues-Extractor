from sklearn.model_selection import train_test_split
import json
from configs.paths_config import VECTORIZED_TRAIN_PATH, VECTORIZED_TEST_PATH
from configs.train_config import TEST_SIZE, RANDOM_STATE


def split_train_test(encoded_issues_scores: list[dict[str, list[int]]]) -> None:
    """Splits encoded issues and scores into train and test sets, and saves them."""
    train, test = train_test_split(encoded_issues_scores, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    with open(VECTORIZED_TRAIN_PATH, "w") as f:
        for row in train:
            f.write(json.dumps(row) + "\n")

    print(f"Train set saved to {VECTORIZED_TRAIN_PATH} with {len(train)} samples.")

    with open(VECTORIZED_TEST_PATH, "w") as f:
        for row in test:
            f.write(json.dumps(row) + "\n")

    print(f"Test set saved to {VECTORIZED_TEST_PATH} with {len(test)} samples.")
