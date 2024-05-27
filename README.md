# Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?
This is the official repository of "Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?".

This work explores the current state of causal reasoning capabilities in large language models (LLMs) and introduces CausalProbe 2024, a new benchmark to assess their performance on fresh and unseen data. Our findings suggest that while LLMs demonstrate understanding of contextual causality, they primarily engage in shallow (level-1) causal reasoning due to the embedded causal knowledge in their parameters, lacking the capacity for genuine human-like (level-2) causal reasoning. We delve into the autoregression mechanism of transformer-based LLMs, revealing its non-causal nature, and provide empirical results showing a significant performance drop on CausalProbe 2024 compared to earlier benchmarks. This repository serves as a starting point for researchers interested in advancing LLMs towards genuine causal reasoning, providing the necessary code, data, and instructions to reproduce our results and encourage further exploration.

![overview_diagram](https://github.com/Haoang97/CausalProbe-2024/blob/main/images/overview.jpg "Overview of this work.")

# Content
- [Installation](#installation)
- [Data](#data)

# Installation
Install dependent Python libraries by running the command below.
```
pip install -r requirements.txt
```
You can also create a conda environment by running the command below.
```
conda env create -f environment.yml
```

# Data
## Causal Q&A benchmarks
Our proposed CausalProbe 2024 consists of two sub-datasets: CausalProbe 2024 Easy (CausalProbe-E) and CausalProbe 2024 Hard (CausalProbe-H), which are formatted in `.json` files. We upload them to the folder `/benchmarks/CausalProbe_2024/`.

For user convenience, we also upload the (pre-propossed) other benchmarks used in this work to the folder `/benchmarks/`, including [COPA](https://people.ict.usc.edu/~gordon/copa.html), [e-CARE](https://github.com/Waste-Wood/e-CARE), and [CausalNet](https://anonymous.4open.science/r/causal-reasoning-0B6E/). For ease of use, we also convert these three benchmarks into `.json` or `.jsonl` format.

## Retrieval document
The retrieval document used in this work is a general knowledge Q&A dataset ([Link](https://huggingface.co/datasets/MuskumPillerum/General-Knowledge)). You can download it by (firstly, make sure the `huggingface_hub` package is installed; if not, run `pip install -U huggingface_hub`)
```
huggingface-cli download --repo-type dataset --resume-download MuskumPillerum/General-Knowledge --local-dir 'Your local dir' --local-dir-use-symlinks False
```
Each data item of general knowledge Q&A is a question-answer pair, like:
```
{"id": 1,
"Question": "What is Artificial Intelligence?",
"Answer": "Artificial Intelligence refers to the development of computer systems that can perform tasks that would typically require human intelligence, such as visual perception, speech recognition, decision-making, and language translation.\\n"
}
```
As a retrieval knowledge base, we only use the answer part of general knowledge Q&A dataset. The answer part covers most of the knowledge in the question part, so the information loss is negligible.

Due to the network limitation, we choose a local and small knowledge base as our retrieval document. In the future, we will further explore to use search engines (like Wikipedia) as external knowledge bases.

# Evaluation
