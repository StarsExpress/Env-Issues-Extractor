from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import json
import torch
import re
from configs.prompts_config import EXTRACTION_PROMPT, MODEL_ID, MAX_NEW_TOKENS


def init_gen_pipeline():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
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


def extract_issues(articles: list[str], gen_pipeline) -> list[dict[str, int]]:
    issues2scores_list: list[dict[str, int]] = []

    for idx, article in enumerate(articles):
        prompt = EXTRACTION_PROMPT.format(article_text=article)
        output = gen_pipeline(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0
        )[0]["generated_text"]

        print(f"Article {idx} raw out: {output}.")

        try:
            issues2scores = json.loads(output)
            issues2scores_list.append(issues2scores)

        except json.JSONDecodeError:
            matches = re.findall(r"\{.*?\}", output, re.S)
            issues2scores_list.extend(json.loads(m) for m in matches if m)

    return issues2scores_list
