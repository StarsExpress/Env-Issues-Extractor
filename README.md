# 🌏 Environmental Issues Extractor

## 📑 Project Description
Supervised fine-tuned LLM that detects and quantifies environment issues from 10,000+ environmental article titles.

## 📦 Distillation LLMs
### 👩🏻‍🏫 Selected Teacher: Qwen/Qwen3-4B-Instruct-2507
**https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507**

![qwen_4b_overview.png](images/qwen_4b_overview.png)

Send 10,000+ environmental article titles to the teacher LLM,
letting it extract and quantify environmental issues and save all results.

### 👨🏻‍🎓 Selected Student: Qwen/Qwen2.5-3B-Instruct
**https://huggingface.co/Qwen/Qwen2.5-3B-Instruct**

![qwen_3b_overview.png](images/qwen_3b_overview.png)

Split 80% of the teacher's extracted and quantified issues to fine-tune the student LLM,
and 20% for the evaluation on student LLM.

## 📊 Evaluation Metrics
#### MAE: 0.0038806808544477205
#### RMSE: 0.029841735729631162
#### Total samples: 2187

## 📲 Example Issues Extraction

```json
{
  "title": "Carbon emissions from England's roads plan '100 times greater than government claims'",
  "issues": {
    "air pollution": 1,
    "biodiversity loss": 10,
    "climate change": 9,
    "energy crisis": 1,
    "fossil fuel dependency": 8,
    "waste": 1
  }
}
```