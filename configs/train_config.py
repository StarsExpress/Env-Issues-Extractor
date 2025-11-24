
# Train test split.
TEST_SIZE = 0.2
RANDOM_STATE = 42


# LLM selection.
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"


# LoRA config.
RANK = 32
ALPHA = 64
DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj"]


# SFT config.
TRAIN_BATCH_SIZE = 4  # Don't go too high to prevent OOM.
EPOCHS = 2
LEARNING_RATE = 2e-5
MAX_SEQ_LEN = 512
