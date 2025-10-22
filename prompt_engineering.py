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
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    gen_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto"
    )
    print("Generation pipeline initialized.")
    return gen_pipeline


def extract_issues(articles: list[str], gen_pipeline) -> list[tuple[str, dict[str, int]]]:
    articles_issues2scores_list: list[tuple[str, dict[str, int]]] = []

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
            articles_issues2scores_list.append((article, issues2scores))

        except json.JSONDecodeError:
            matches = re.finditer(r"\{[^{}]+\}", output, flags=re.DOTALL)
            for match in matches:
                try:
                    obj = json.loads(match.group())
                    articles_issues2scores_list.append((article, obj))

                except json.JSONDecodeError:
                    continue

        if len(articles_issues2scores_list) >= 10:
            break

    return articles_issues2scores_list
