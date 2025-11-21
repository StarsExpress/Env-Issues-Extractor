from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
import torch
from configs.paths_config import SFT_TRAIN_PATH, MODEL_FOLDER
from configs.train_config import MODEL_ID


def train_llm():
    # -----------------------------------------------
    # Tokenizer.
    # -----------------------------------------------
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------
    # Dataset.
    # -----------------------------------------------
    print("Loading SFT train dataset...")
    dataset = load_dataset(
        "json",
        data_files=SFT_TRAIN_PATH,
        split="train",
    )

    # -----------------------------------------------
    # Base LLM.
    # -----------------------------------------------
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # -----------------------------------------------
    # LoRA configuration.
    # -----------------------------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    # -----------------------------------------------
    # SFT Configuration (REPLACES TrainingArguments).
    # -----------------------------------------------
    sft_config = SFTConfig(
        output_dir=MODEL_FOLDER,
        num_train_epochs=3,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        logging_steps=20,
        save_steps=500,
        save_total_limit=1,
        bf16=True,
    )

    # -----------------------------------------------
    # Trainer.
    # -----------------------------------------------
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="messages",  # OK in TRL 0.9.3
        peft_config=lora_config,
        args=sft_config  # MUST BE SFTConfig.
    )

    # -----------------------------------------------
    # Train.
    # -----------------------------------------------
    print("Training starts...")
    trainer.train()

    # -----------------------------------------------
    # Save finetuned LLM.
    # -----------------------------------------------
    print("Saving finetuned LLM...")
    trainer.save_model(MODEL_FOLDER)
    tokenizer.save_pretrained(MODEL_FOLDER)

    print("Training complete! Finetuned LLM saved to:", MODEL_FOLDER)
