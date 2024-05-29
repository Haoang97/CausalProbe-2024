import os
import argparse
import numpy as np
from tqdm import tqdm
import random
import argparse
import torch
#from vllm import LLM, SamplingParams
from utils import load_file, TASK_INST, save_file_jsonl, process_arc_instruction, postprocess_answers_closed, clean_dict_list
from prompts import PROMPT_DICT
from metrics import metric_max_over_ground_truths, exact_match_score, match, binary_choice_match, multiple_choice_match
from passage_retrieval import Retriever
import ast
import backoff
import openai
from openai.error import APIError, Timeout, APIConnectionError
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights,infer_auto_device_map,load_checkpoint_in_model,dispatch_model

ORG_KEY="YOUR_ORG_KEY"
#os.environ["CUDA_VISIBLE_DEVICES"] = "3" 

@backoff.on_exception(backoff.expo, openai.error.RateLimitError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def completions_instructgpt_backoff(**kwargs):
    return openai.Completion.create(**kwargs)


def call_model_chatgpt(prompt, model, max_tokens=50):
    print(model)
    try:
        results = completions_with_backoff(
            model=model,
            messages=[
                {"role": "user",
                    "content": prompt},
            ],
            request_timeout=5,
            max_tokens=max_tokens,
        )
        result = results["choices"][0]["message"]["content"]
    except (APIError, Timeout, APIConnectionError):
        result = "ERROR: API error outputs"
    return result

def call_model_instructgpt(prompt, model, max_tokens=50):
    try:
        results = completions_instructgpt_backoff(model=model, prompt=prompt, temperature=0.0,
                                                  max_tokens=max_tokens, logprobs=5, top_p=1, frequency_penalty=0.0, presence_penalty=0.0)
        result = results["choices"][0]["text"]
    except (APIError, Timeout, APIConnectionError):
        results = "ERROR: API error outputs"
    return result


def call_model(prompts, model, tokenizer, max_new_tokens=50):
    # using vllm package:
    #sampling_params = SamplingParams(
    #    temperature=0.8, top_p=0.95, max_tokens=max_new_tokens)
    #preds = model.generate(prompts, sampling_params)
    model_name = model.name_or_path.split("/")[-1]
    tokens = tokenizer.batch_encode_plus(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        if model_name == "llama-3-8b-instruct":
            terminators = [
                tokenizer.eos_token_id,
                tokenizer.convert_tokens_to_ids("<|eot_id|>")
            ]
            preds = model.generate(**tokens, max_new_tokens=max_new_tokens, eos_token_id=terminators,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id)
        else:
            preds = model.generate(**tokens, max_new_tokens=max_new_tokens,
                                do_sample=False, pad_token_id=tokenizer.pad_token_id)
    preds = tokenizer.batch_decode(preds[:, tokens.input_ids.shape[1]:], skip_special_tokens=True)
    #preds = [pred.outputs[0].text.split("\n\n")[0] for pred in preds]
    postprocessed_preds = [postprocess_output(pred) for pred in preds]
    return postprocessed_preds, preds


def postprocess_output(pred):
    pred = pred.replace("</s>", "")
    pred = pred.replace("\n", "") # Qwen 14B chat tends to generate "\n" at beginning

    if len(pred) > 0 and pred[0] == " ":
        pred = pred[1:]
    return pred

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str,
                        default="/home/user/chihaoang/llama/models_hf/7Bf")
    parser.add_argument('--retriever_path', type=str,
                        default="/home/user/chihaoang/self-rag-main/facebook/contriever-msmarco")
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--passages', type=str, default=None)
    parser.add_argument('--passages_embeddings', type=str, default=None)
    parser.add_argument('--passages_source', type=str, default=None)
    parser.add_argument('--mode', type=str, default="vanilla")
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--int8bit', action="store_true")
    parser.add_argument("--no_fp16", action="store_true", help="retriever inference in fp32")
    parser.add_argument("--projection_size", type=int, default=768, help="vector size in retriever")
    parser.add_argument(
        "--n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument("--n_bits", type=int, default=8, help="Number of bits per subquantizer")
    parser.add_argument(
        "--indexing_batch_size", type=int, default=1000000, help="Batch size of the number of passages indexed"
    )
    parser.add_argument(
        "--save_or_load_index", action="store_true", help="If enabled, save index and load index if it exists"
    )
    parser.add_argument("--lowercase", action="store_true", help="lowercase text before encoding")
    parser.add_argument("--normalize_text", action="store_true", help="normalize text")
    parser.add_argument("--question_maxlength", type=int, default=512, help="Maximum number of tokens in a question")
    parser.add_argument("--per_gpu_batch_size", type=int, default=64, help="Batch size for question encoding")
    parser.add_argument("--n_docs", type=int, default=20, help="Number of documents to retrieve per questions")
    parser.add_argument("--top_n", type=int, default=5, help="Number of documents to be used, top_n<=n_docs")
    parser.add_argument("--load_retrieved_docs", type=eval, default=True, help="Whether load pre-retrieved docs or not")
    parser.add_argument('--metric', type=str)
    parser.add_argument('--result_fp_base', type=str)
    parser.add_argument('--task', type=str)
    parser.add_argument('--prompt_name', type=str, default="prompt_no_input")
    parser.add_argument('--batch_size', type=int, default=5)
    parser.add_argument("--dtype",  type=str, default=None,
                        help="parameter type when loading LLMs")
    parser.add_argument("--world_size",  type=int, default=1,
                        help="world size to use multiple GPUs.")
    parser.add_argument("--choices",  type=str, default=None,
                        help="space-separated answer candidates")
    parser.add_argument("--instruction",  type=str,
                        default=None, help="task instructions")
    parser.add_argument('--download_dir', type=str, help="specify download dir",
                        default=".cache")
    parser.add_argument('--api_key', type=str, default=None)
    parser.add_argument('--api_base', type=str, default=None)
    args = parser.parse_args()

    result_fp = args.result_fp_base + f"result_{(args.model_name.split('/'))[-1]}_{args.mode}_{args.input_file.split('/')[-1].split('.')[0]}_{args.prompt_name[7:]}.json"
    print(f"Results saved in {result_fp}")

    isOpenAI = True if any(sub in args.model_name for sub in ["gpt", "davinci"]) else False

    if isOpenAI is False:
        if args.dtype is not None:
            #model = LLM(model=args.model_name, download_dir=args.download_dir, dtype=args.dtype,
            #            tensor_parallel_size=args.world_size,)
            model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", torch_dtype=args.dtype).eval()
        else:
            #model = LLM(model=args.model_name, download_dir=args.download_dir,
            #            tensor_parallel_size=args.world_size,)
            model = AutoModelForCausalLM.from_pretrained(args.model_name, device_map="auto", trust_remote_code=True).eval()
        if args.model_name.split("/")[-1].split("-")[0] == "Qwen":
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", trust_remote_code=True, pad_token="<|endoftext|>")
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.model_name, padding_side="left", trust_remote_code=True)
            tokenizer.pad_token = tokenizer.eos_token

    input_data = load_file(args.input_file)

    if isOpenAI is True and args.api_key is not None:
        openai.api_key = args.api_key
        openai.api_base = args.api_base

    # For baseline scripts, we simply load pre-retrieved documents from `passages` option.
    if args.mode == "retrieval":
        # Set a retriever
        if args.load_retrieved_docs:
            retrieved_results = load_file(f"./retrieved_docs/{args.passages_source}_{args.input_file.split('/')[-1].split('.')[0]}_20.json")
            print("Retrieved docs loading done.")
        else:
            retriever = Retriever(args)
            retriever.setup_retriever()
            retrieved_results = retriever.search_document(input_data, args.n_docs)
        '''
        if args.passages_source == "general_knowledge":
            retrieved_results = [
                {
                    "id": line["id"],
                    "passages": "\n\n".join([item["passage"] for item in line["passages"][:args.top_n]])
                }
                for line in retrieved_results
            ]

            id2retrieval = {}
            for id, item in enumerate(retrieved_results):
                if input_data[id]["id"] == item["id"]:
                    id2retrieval[input_data[id]["qid"]] = item["passages"]

            for item in input_data:
                item["paragraph"] = id2retrieval[item["id"]]
         
        elif args.passages_source == "wikipedia":
        '''
        id2retrieval = {}
        for item in retrieved_results:
            if args.passages_source == "wikipedia":
                id2retrieval[item["id"]] = [i["title"] + "\n" + i["text"] for i in item["passages"][:args.top_n]]
            elif args.passages_source == "general_knowledge":
                id2retrieval[item["id"]] = [i["passages"].strip() for i in item["passages"][:args.top_n]]
            else:
                NotImplementedError
        for id, item in enumerate(input_data):
            #retrieved_results = id2retrieval[id if "id" not in item else item["id"]]
            #evidences = ["[{}] ".format(
            #    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieved_results)]
            evidences = id2retrieval[item["id" if "id" in item else "index"]]
            item["paragraph"] = "\n".join(evidences)
        #else:
        #    NotImplementedError
        
        del retrieved_results
        '''
        if args.passages is not None:
            retrieval_data = load_file(args.passages)
            id2retrieval = {}
            for id, item in enumerate(retrieval_data):
                if "id" not in item:
                    #id2retrieval[id] = item["ctxs"][:args.top_n]
                    id2retrieval[id] = item["Question"] + "\n" + item["Answer"]
                else:
                    id2retrieval[item["id"]] = item["Question"] + "\n" + item["Answer"]
            for id, item in enumerate(input_data):
                retrieval_result = id2retrieval[id if "id" not in item else item["id"]]
                evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)
        else:
            for id, item in enumerate(input_data):
                retrieval_result = item["ctxs"][:args.top_n]
                evidences = ["[{}] ".format(
                    i+1) + ctx["title"]+"\n" + ctx["text"] for i, ctx in enumerate(retrieval_result)]
                item["paragraph"] = "\n".join(evidences)
        '''

    for item in input_data:
        if "golds" not in item:
            if "output" in item:
                item["golds"] = item["output"]
            if "answers" in item:
                item["golds"] = item["answers"]
            if "answer" in item:
                item["golds"] = item["answer"]
            if "possible_answers" in item:
                item["golds"] = ast.literal_eval(item["possible_answers"])
            if "answerKey" in item:
                item["golds"] = [item["answerKey"]]
            if "label" in item:
                item["golds"] = item["label"]
            if "Correct_answer" in item:
                item["golds"] = item["Correct_answer"]

        if "instruction" not in item and "question" in item:
            item["instruction"] = item["question"]

        if args.instruction is not None:
            item["instruction"] = args.instruction + \
                "\n\n### Input:\n" + item["instruction"]
        if args.task == "fever" or args.task == "arc_c":
            item["instruction"] = TASK_INST[args.task] + \
                "\n\n### Input:\n" + item["instruction"]

    final_results = []
    for idx in tqdm(range(len(input_data) // args.batch_size)):
        batch = input_data[idx*args.batch_size:(idx+1)*args.batch_size]
        processed_batch = [
            PROMPT_DICT[args.prompt_name].format_map(item) for item in batch]

        if isOpenAI is True:
            preds = []
            for input_instance in processed_batch:
                if args.model_name == "text-davinci-003":
                    pred = call_model_instructgpt(
                        input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                if args.model_name == "gpt-3.5-turbo-0301" or args.model_name == "gpt-3.5-turbo":
                    pred = call_model_chatgpt(
                        input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                preds.append(pred)
        else:
            preds, _ = call_model(
                processed_batch, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            pred = preds[j]
            item["output"] = postprocess_answers_closed(
                pred, args.task, args.choices)
            item["output"] = pred
            final_results.append(item)

    if len(input_data) % args.batch_size > 0:
        batch = input_data[(idx+1)*args.batch_size:]
        processed_batch = [
            PROMPT_DICT[args.prompt_name].format_map(item) for item in batch]
        if isOpenAI is True:
            preds = []
            for input_instance in processed_batch:
                if args.model_name == "text-davinci-003":
                    pred = call_model_instructgpt(
                        input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                if args.model_name == "gpt-3.5-turbo-0301" or args.model_name == "gpt-3.5-turbo":
                    pred = call_model_chatgpt(
                        input_instance, model=args.model_name, max_tokens=args.max_new_tokens)
                preds.append(pred)
        else:
            preds, _ = call_model(
                processed_batch, model=model, tokenizer=tokenizer, max_new_tokens=args.max_new_tokens)
        for j, item in enumerate(batch):
            pred = preds[j]
            item["output"] = postprocess_answers_closed(
                pred, args.task, args.choices)
            final_results.append(item)

    for item in input_data:
        if args.metric == "em":
            metric_result = metric_max_over_ground_truths(
                exact_match_score, item["output"], item["golds"])
        elif args.metric == "accuracy":
            metric_result = 1.0 if item["golds"][0] in item["output"] else 0.0
        elif args.metric == "match":
            metric_result = match(item["output"], item["golds"])
        elif args.metric == "exact_match_score":
            metric_result = binary_choice_match(item["output"], item["golds"])
        elif args.metric == "binary_choice_match":
            metric_result = binary_choice_match(item["output"], item["golds"])
        elif args.metric == "multiple_choice_match":
            metric_result = multiple_choice_match(item["output"], item["golds"])
        else:
            raise NotImplementedError
        item["metric_result"] = metric_result

    all_results = [item["metric_result"] for item in input_data]
    vaild_results = [item for item in all_results if isinstance(item, bool)] # the boolean results, or outputs cotains choice id
    adjusted_results = [item if isinstance(item, bool) else False for item in all_results] # replace non-boolean results with False
    print(f"vaild answers/all: {len(vaild_results)}/{len(all_results)}")
    print("overall exact match: {0}".format(
        np.mean(adjusted_results)))
    print("valid exact match: {0}".format(
        np.mean(vaild_results)))

    if args.task == "factscore":
        processed_item = []
        for item in input_data:
            processed_item.append(item)
        save_file_jsonl(processed_item, result_fp)
    else:
        input_data = [{key: value for key, value in item.items() if key not in ["golds","instruction"]} for item in input_data]
        save_file_jsonl(input_data, result_fp)


if __name__ == "__main__":
    main()