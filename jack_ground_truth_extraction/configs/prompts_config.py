
EXTRACTION_PROMPT = """
You are an environmental news analyst. 
Your task is to extract key environmental issues mentioned in the following article 
(e.g., air pollution, waste, energy crisis, biodiversity, etc.). 

For each relevant issue, assign a severity score from 1 (minimal concern) to 10 (critical concern)
based on the article’s tone and facts.

Use the following heuristics for severity:
1–3: minor mentions or speculative discussion
4–6: moderate concern or localized problem
7–8: major environmental impact or crisis discussed
9–10: catastrophic or urgent threat with strong emphasis

Respond only with a valid JSON dictionary, with no extra text or commentary.
Do not add brackets, parentheses, quotes, or outside formatting for any issue.

### Example:
{{
  "climate change": 8,
  "air pollution": 6,
  "deforestation": 4
}}

### Article:
{article_text}
"""


MODEL_ID = "openai/gpt-oss-20b"

MAX_NEW_TOKENS = 256
