# Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?
This is the official repository of "Unveiling Causal Reasoning in Large Language Models: Reality or Mirage?".

This work explores the current state of causal reasoning capabilities in large language models (LLMs) and introduces CausalProbe 2024, a new benchmark to assess their performance on fresh and unseen data. Our findings suggest that while LLMs demonstrate understanding of contextual causality, they primarily engage in shallow (level-1) causal reasoning due to the embedded causal knowledge in their parameters, lacking the capacity for genuine human-like (level-2) causal reasoning. We delve into the autoregression mechanism of transformer-based LLMs, revealing its non-causal nature, and provide empirical results showing a significant performance drop on CausalProbe 2024 compared to earlier benchmarks. This repository serves as a starting point for researchers interested in advancing LLMs towards genuine causal reasoning, providing the necessary code, data, and instructions to reproduce our results and encourage further exploration.

![overview_diagram](https://github.com/Haoang97/CausalProbe-2024/blob/main/images/overview.jpg "Overview of this work.")

# Content
- [Installation](#installation)
- [Data](#data)
   * [Causal Q&A benchmarks](#causal-qa-benchmarks)
   * [Retrieval document](#retrieval-document)
- [Inference](#inference)
   * [Vanilla inference](#vanilla-inference)
   * [Chain-of-Thought (CoT)](#chain-of-thought-cot)
   * [Retrieval-augmented Generation (RAG)](#retrieval-augmented-generation-rag)
   * [G^2-Reasoner](#g2-reasoner)
- [Reference](#reference)
- [Contact](#contact)

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

# Inference
Our code mainly follows the realization of vanilla RAG in [Self-RAG](https://github.com/AkariAsai/self-rag/tree/main). We use [Contriever-MSMARCO](https://github.com/facebookresearch/contriever) as the retrieval module.
## Vanilla inference
The `gpu_ids` mainly depends on the LLM's size and batch size. The value of `prompt_name` depends on the used benchmark and reasoning method. The full list of prompts is shown in `prompts.py` and you can just choose your needed prompt from this file.
```
CUDA_VISIBLE_DEVICES=[gpu_ids] python main.py \
    --model_name [LLM dir or name] \
    --input_file [benchmark dir] \
    --mode vanilla \
    --batch_size 16 \
    --max_new_tokens 50 \
    --metric multiple_choice_match \
    --prompt_name "prompt_mcqa_[benchmark_name]" \
    --task qa \
    --result_fp_base ./result_logs/ \
    --api_key [your API key if using closed-source models] \
    --api_base [your API's base url if necessary]
```

## Chain-of-Thought (CoT)
```
CUDA_VISIBLE_DEVICES=[gpu_ids] python main.py \
    --model_name [LLM dir or name] \
    --input_file [benchmark dir] \
    --mode vanilla \
    --batch_size 16 \
    --max_new_tokens 128 \
    --metric multiple_choice_match \
    --prompt_name "prompt_mcqa_cot_[benchmark_name]" \
    --task qa \
    --result_fp_base ./result_logs/ \
    --api_key [your API key if using closed-source models] \
    --api_base [your API's base url if necessary]
```

## Retrieval-augmented Generation (RAG)
First, we should construct a vector database using the`faiss` package. Given a retrieval knowledge base, we generate its embeddings following the Self-RAG repository.
```
for i in 0
do
  export CUDA_VISIBLE_DEVICES=$i
  python generate_embeddings.py \
    --model_name_or_path [contriever-msmarco or your embedding model dir] \
    --output_dir [your output dir] \
    --passages [your retrieval knowledge base dir] \
    --shard_id $i \
    --num_shards 1 > ./nohup.my_embeddings.$i 2>&1 &
done
```
Then, with the original retrieval knowledge base, its embeddings, and a retriever, we can do RAG. You need to retrieve a knowledge base the first time you use it. In the following, you can set the `--load_retrieved_docs=True` to use the saved retrieved results. We have uploaded the retrieved results in `./retrieved_docs/`. We retrieve and save the Top-20 related knowledge for each  question.
```
CUDA_VISIBLE_DEVICES=[gpu_ids] python rag.py \
    --model_name  [LLM dir or name] \
    --input_file [benchmark dir] \
    --passages [retrieval knowledge base dir] \
    --passages_embeddings [retrieval knowledge embeddings dir] \
    --passages_source [knowledge embeddings name] \
    --retriever_path [retriever dir] \
    --mode retrieval \
    --n_docs 20 \
    --top_n 1 \
    --load_retrieved_docs False \
    --batch_size 8 \
    --max_new_tokens 50 \
    --metric multiple_choice_match \
    --prompt_name "prompt_mcqa_retrieval_[benchmark_name]" \
    --task qa \
    --result_fp_base ./result_logs/ \
    --api_key [your API key if using closed-source models] \
    --api_base [your API's base url if necessary]
```
## G^2-Reasoner
Like RAG, you can run the G^2-Reasoner with the following command
```
CUDA_VISIBLE_DEVICES=[gpu_ids] python rag.py \
    --model_name  [LLM dir or name] \
    --input_file [benchmark dir] \
    --passages [retrieval knowledge base dir] \
    --passages_embeddings [retrieval knowledge embeddings dir] \
    --passages_source [knowledge embeddings name] \
    --retriever_path [retriever dir] \
    --mode retrieval \
    --n_docs 20 \
    --top_n 1 \
    --load_retrieved_docs False \
    --batch_size 8 \
    --max_new_tokens 50 \
    --metric multiple_choice_match \
    --prompt_name "prompt_mcqa_g2reasoner_[benchmark_name]" \
    --task qa \
    --result_fp_base ./result_logs/ \
    --api_key [your API key if using closed-source models] \
    --api_base [your API's base url if necessary]
```
# Reference
[1]  Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, and Hannaneh Hajishirzi. SELF-RAG: Learning to Retrieve, Generate and Critique through self-reflection. ICLR 2024

[2] G. Izacard, M. Caron, L. Hosseini, S. Riedel, P. Bojanowski, A. Joulin, E. Grave Unsupervised Dense Information Retrieval with Contrastive Learning

[3] Melissa Roemmele, Cosmin Adrian Bejan, Andrew S. Gordon. Choice of Plausible Alternatives: An Evaluation of Commonsense Causal Reasoning. AAAI 2011.

[4] Li Du, Xiao Ding, Kai Xiong, Ting Liu, Bing Qin. e-CARE: a New Dataset for Exploring Explainable Causal Reasoning. ACL 2022.

[5] Ashwani S, Hegde K, Mannuru N R, et al. Cause and Effect: Can Large Language Models Truly Understand Causality?[J]. arXiv preprint arXiv:2402.18139, 2024.

# Contact
Due to the double-blind requirement, we do not provide our e-mails during submission.
