from transformers import AutoTokenizer, AutoModelForCausalLM
from configs.paths_config import MODEL_FOLDER
from evaluator import load_student_model, run_student_inference, evaluate_llm
from trainer import train_llm


def main() -> None:
    train_llm()

    # print("Load fine-tuned student model.")
    # tokenizer = AutoTokenizer.from_pretrained(MODEL_FOLDER)
    # model = AutoModelForCausalLM.from_pretrained(MODEL_FOLDER)

    # model, tokenizer = load_student_model()
    # run_student_inference(model, tokenizer)
    # evaluate_llm()


if __name__ == "__main__":
    main()
