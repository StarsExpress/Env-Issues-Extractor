
# Train test split.
TEST_SIZE = 0.2
RANDOM_STATE = 42


# LLM selection.
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


# LoRA config.
ATTENTION_DIM = 8
DROPOUT = 0.05


# SFT config.
TRAIN_BATCH_SIZE = 1  # Prevent OOM because training prompts are long.
EPOCHS = 2
LEARNING_RATE = 2e-4
MAX_SEQ_LEN = 512
