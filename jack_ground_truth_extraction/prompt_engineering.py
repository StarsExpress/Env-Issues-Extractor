from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import torch
import re
from configs.prompts_config import EXTRACTION_PROMPT, MODEL_ID, MAX_NEW_TOKENS, BATCH_SIZE
from utils import append_raw_issues


def init_gen_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        load_in_8bit=False
    )
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    print("Generation pipeline initialized.")
    return gen_pipeline


def extract_issues(articles: list[tuple[str, str]], gen_pipeline) -> None:
    # batch inference
    for start in range(0, len(articles), BATCH_SIZE):
        batch = articles[start : start + BATCH_SIZE]
        titles, texts = zip(*batch)
        prompts = [EXTRACTION_PROMPT.format(article_text=t) for t in texts]

        outputs = gen_pipeline(
            prompts,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0
        )

        batch_dicts = []
        for idx, out_list in enumerate(outputs, start=start):
            # pipeline returns list[dict] per input even when num_return_sequences=1
            text = out_list[0]["generated_text"]
            print(f"Article {idx} raw out: {text}.")
            matches = re.findall(r"\{.*?\}", text, re.S)
            if matches:
                # take the last (assumed most complete) dict
                issues_json = matches[-1]
            else:
                # fallback: attempt to load entire text
                issues_json = text

            try:
                issues_dict = json.loads(issues_json)
                batch_dicts.append({"title": titles[idx-start], "issues": issues_dict})
            except json.JSONDecodeError:
                print(f"Warning: could not parse JSON for article {titles[idx-start]}")

        append_raw_issues(batch_dicts)

    # nothing returned; data streamed
