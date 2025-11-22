from trainer import train_llm
from evaluator import evaluate_llm


def main() -> None:
    train_llm()
    evaluate_llm()


if __name__ == "__main__":
    main()
