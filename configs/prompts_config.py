EXTRACTION_PROMPT = """
You are an environmental news analyst.

First, briefly list the environmental issues you identify (1 sentence each).
Then, on the next line, output ONLY the JSON dictionary of issues and their 1–10 scores.

Example of JSON format (do NOT copy values):
{{"<issue>": <score>, ...}}

Article:
{article_text}
"""

MODEL_ID = "openai/gpt-oss-20b"

MAX_NEW_TOKENS = 256
