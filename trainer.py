import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch
from configs.paths_config import SFT_TRAIN_PATH, MODEL_FOLDER
from configs.train_config import MODEL_ID


# Correct Qwen-compatible formatting function.
def build_formatter(tokenizer):
    def formatting_func(example):
        messages = example["messages"]

        # Qwen requires using its own chat template.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
        return text

    return formatting_func


def train_llm():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading SFT train dataset...")
    # ---- Avoid Arrow / TMP writing completely. ----
    data_list = []

    with open(SFT_TRAIN_PATH, "r") as f:
        for line in f:
            data_list.append(json.loads(line))

    dataset = Dataset.from_list(data_list)

    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="eager"
    )

    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    sft_config = SFTConfig(
        output_dir=MODEL_FOLDER,
        num_train_epochs=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=20,
        save_steps=500,
        save_total_limit=1,
        bf16=True,
        packing=False,
        max_seq_length=512,
    )

    # Clean bad samples.
    clean_samples = [
        row for row in dataset
        if isinstance(row.get("messages", []), list) and len(row["messages"]) >= 2
    ]
    dataset = Dataset.from_list(clean_samples)
    print("Clean samples size:", len(dataset))

    # Build correct formatting function.
    formatting_func = build_formatter(tokenizer)

    print("Setting up trainer...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,
        dataset_text_field=None,
        peft_config=lora_config,
        args=sft_config,
    )

    print("Training starts...")
    trainer.train()

    print("Saving finetuned LLM...")
    trainer.save_model(MODEL_FOLDER)
    tokenizer.save_pretrained(MODEL_FOLDER)

    print("Training complete! Model saved to:", MODEL_FOLDER)
