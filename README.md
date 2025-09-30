# Socratic Self-Refine (SSR)
This is the official repo for the paper "SSR: Socratic Self-Refine for Large Language Model Reasoning" 

## Environment Setup
```sh
pip install openai ipdb math_verify datasets huggingface
```

## ⚙️ API Configuration Setup

You need to set up your API key and URL:

1. If you would like to use the OpenAI model, create an `apikey.py` file in the project root directory with the following format:
```python
url = "https://api.openai.com/v1"  # Replace with your API endpoint
openai_api_key = [
    "your-api-key-here",  # Replace with your actual API key
    # You can add multiple API keys to improve concurrency performance.
]
```
2. If you want to use the Gemini model, locate the following code snippet in `module_atomic.py` and fill in your corresponding information: 
```python
# TODO(developer): Update and un-comment below lines
project_id = "<your-project-id>"
location = "<your-location>"
```


## 🚀 Quick Start
### Running the Experiments
```bash
bash scripts/gpt-5-mini.sh
```

### Running the Evaluation (result extraction)
Please refer to `eval.ipynb`.

## 📋 Implementation
### 🤔 Socratic Self-Refine Variants (Ours)
- [x] SSR-Lin
- [x] SSR-Ada
- [x] SSR-Plan

### 🤖 Baselines
- [x] Chain-of-Thoughts (CoT)
- [x] Self-Refine
- [x] Debate
- [x] Chain-of-Thoughts Self-Consistency (CoT-SC)
- [x] Monte-Carlo Tree Self-refine (MTCSr), to replace ToT (a bit outdated)
- [x] Atom-of-Thoughts (AoT)

### 📈 Datasets
- [x] MATH-Level-5
- [x] AIME24
- [x] AIME25
- [x] Zebra-Puzzles
- [x] Mini-Sudoku