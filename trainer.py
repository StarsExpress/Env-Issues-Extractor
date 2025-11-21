from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch
from configs.paths_config import SFT_TRAIN_PATH, MODEL_FOLDER
from configs.train_config import MODEL_ID


def formatting_func(example) -> list[str]:
    messages = example.get("messages", [])
    if not isinstance(messages, list) or len(messages) == 0:
        return ["User: dummy\nAssistant: dummy\n"]

    text = ""
    for message in messages:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user").capitalize()
        content = message.get("content", "")
        text += f"{role}: {content}\n"

    text = text.strip()
    if len(text) > 1600:  # Ensure memory efficiency.
        text = text[:1600].rstrip()

    if len(text) == 0:
        text = "User: dummy\nAssistant: dummy\n"

    return [text]  # Must return a list of strings for the trainer.


def train_llm():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading SFT train dataset...")
    dataset = load_dataset(
        "json",
        data_files=SFT_TRAIN_PATH,
        split="train",
    )

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
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj"
        ]
    )

    sft_config = SFTConfig(
        output_dir=MODEL_FOLDER,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        logging_steps=20,
        save_steps=500,
        save_total_limit=1,
        bf16=True,
        packing=False,  # 🚨 MUST BE DISABLED.
        max_seq_length=512,  # Ensure seq length is manageable for memory.
    )

    print("Filtering bad samples...")
    clean_samples = []

    for row in dataset:
        messages = row.get("messages", [])
        if isinstance(messages, list) and len(messages) >= 1:
            clean_samples.append(row)

    dataset = Dataset.from_list(clean_samples)  # Rebuild into Dataset.
    print("Clean samples size:", len(dataset))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        formatting_func=formatting_func,  # Must self define for custom formatting.
        dataset_text_field=None,
        peft_config=lora_config,
        args=sft_config,
    )

    print("Training starts...")
    trainer.train()

    print("Saving finetuned LLM...")
    trainer.save_model(MODEL_FOLDER)
    tokenizer.save_pretrained(MODEL_FOLDER)

    print("Training complete! Finetuned LLM saved to:", MODEL_FOLDER)
