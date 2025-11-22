# Environmental Issues Extractor ☃︎

## 📑 Description
Supervised fine-tuned LLM that detects and quantifies environment issues from 10,000+ environmental article titles.

## 📦 Selected LLM
### Qwen/Qwen2.5-3B-Instruct
**https://huggingface.co/Qwen/Qwen2.5-3B-Instruct**

## 📊 Evaluation Metrics
#### MAE: 0.0038806808544477205
#### RMSE: 0.029841735729631162
#### Total samples: 2187

## 📲 Example Issues Extraction

```json
{
  "title": "Thames Water fined \u00a32.3m for raw sewage pollution incident",
  "issues": {
    "bird activity": 8,
    "climate change": 9,
    "floods": 10,
    "water pollution": 7
  }
}
```