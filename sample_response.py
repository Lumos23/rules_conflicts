import os
import json
import argparse
import random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

from transformers.utils import logging
logging.set_verbosity_error()

from tqdm import tqdm

import sys
sys.path.append("/scratch/gpfs/lh2046/rules_conflicts/prompt_gpt")
from prompt_gpt import initialize, DEFAULT_MODEL, get_response_with_retries

def local_vllm_wrapper(
    system_prompt,
    user_prompt,
    past_interactions=[],
    model=None,
    tokenizer=None,
    temperature=0.8,
    max_tokens=2048,
    **kwargs
):
    if kwargs.get("no_system_role", False):
        if system_prompt != "":
            user_prompt = system_prompt + "\n\n" + user_prompt
        chat = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    else:
        chat = [
            {
                "role": "system",
                "content": system_prompt
            }
        ] + past_interactions + [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    
    n_explicitly_passed = "n" in kwargs
    
    formatted = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens, n=kwargs.get("n", 1))
    
    outputs = model.generate(formatted, sampling_params=sampling_params, use_tqdm=False)
    ret = []
    for output in outputs[0].outputs:
        output = output.text
        if "<|eot_id|>" in output:
            output = output[:output.index("<|eot_id|>")]
        output = output.strip()
        ret.append(output)
    
    return ret if n_explicitly_passed else ret[0]

def openai_api_wrapper(
    system_prompt,
    user_prompt,
    past_interactions=[],
    temperature=0.8,
    **kwargs
):
    response = get_response_with_retries(system_prompt, user_prompt, past_interactions, temperature=temperature)

    if "n" in kwargs:
        assert kwargs["n"] == 1, "OpenAI API only supports n=1"
        return [response]
    else:
        return response

def get_responses(
    example,
    n_samples,
    call_wraper,
    call_kwargs,
    system_prompt="", # specify system prompt in this function if needed
):
    responses = []
    
    batch_size = call_kwargs.pop("batch_size", 1)
    
    for i in range(0, n_samples, batch_size):
        n_this_time = min(batch_size, n_samples - i)
        
        responses.extend(
            call_wraper(
                system_prompt, 
                example["prompt"], 
                **call_kwargs,
                n=n_this_time
            )
        )
    return responses

def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--mode", "-M", choices=["local", "openai"], default="openai")
    parser.add_argument("--model", "-m", type=str, default=None)
    parser.add_argument("--tokenizer", "-tk", type=str, default=None)
    parser.add_argument("--temperature", "-t", type=float, default=0.8)
    parser.add_argument("--tensor_parallel_size", "-tps", type=int, default=1)
    parser.add_argument("--bf16", "-bf16", action="store_true")
    parser.add_argument("--max_tokens", "-mt", type=int, default=2048)
    parser.add_argument("--max_model_length", "-ml", type=int, default=8192)
    parser.add_argument("--data", "-d", default="processed_data/data_mixes/halloweenmix.json")
    parser.add_argument("--num_samples", "-N", type=int, default=1)
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--no_system_role", "-nsr", action="store_true")
    parser.add_argument("--start_index", "-si", type=int, default=None)
    parser.add_argument("--end_index", "-ei", type=int, default=None)
    parser.add_argument("--resume", "-r", action="store_true")
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--verbose", "-v", action="store_true")
    
    args = parser.parse_args()
    
    if args.model is None:
        args.model = DEFAULT_MODEL if args.mode == "openai" else "/scratch/gpfs/DANQIC/ab4197/models/Llama-3.1-70B-Instruct"
    if args.tokenizer is None:
        args.tokenizer = args.model
    
    if args.output is None:
        data_name = os.path.basename(args.data)[:-len(".json")]
        model_name = os.path.basename(args.model)
        args.output = f"/scratch/gpfs/lh2046/rules_conflicts/model_response/{data_name}__m_{model_name}__N_{args.num_samples}__t_{args.temperature}.json"
        if args.start_index is not None:
            args.output = args.output.replace(".json", f"__s_{args.start_index}__e_{args.end_index}.json").replace("/responses/", "/response_shards/")
    
    if args.max_model_length < 0:
        args.max_model_length = None
    
    return args

def main():
    args = parse_args()
    
    if args.mode == "local":
        kwargs = {}
        if args.max_model_length is not None:
            kwargs["max_model_len"] = args.max_model_length
        if args.bf16:
            kwargs["dtype"] = "bfloat16"
        if args.tensor_parallel_size > 1:
            kwargs["tensor_parallel_size"] = args.tensor_parallel_size
        llm = LLM(args.model, trust_remote_code=True, **kwargs)
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        
        call_wrapper = local_vllm_wrapper
        call_kwargs = {
            "model": llm,
            "tokenizer": tokenizer,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "no_system_role": args.no_system_role
        }
    else:
        initialize(args.model, verbose=args.verbose)
        call_wrapper = openai_api_wrapper
        call_kwargs = {"temperature": args.temperature}
    
    outputs = []
    
    if args.resume and os.path.exists(args.output):
        with open(args.output, "r") as f:
            outputs = json.load(f)
    
    data = json.load(open(args.data, "r"))
    
    if args.start_index is not None:
        data = data[args.start_index:args.end_index]
    
    model_name = os.path.basename(args.model)
    
    for i in tqdm(range(len(outputs), len(data))):
        example = data[i]
        if "system_prompt" in example:
            system_prompt = example["system_prompt"]
        else:
            system_prompt = ""
        # if "past_interactions" in example:
        #     past_interactions = example["past_interactions"]
        # else:
        #     past_interactions = []
        responses = get_responses(example, args.num_samples, call_wrapper, call_kwargs, system_prompt=system_prompt)
        
        assert len(responses) == args.num_samples, f"Expected {args.num_samples} responses, got {len(responses)}"
        
        outputs.append({
            "prompt": example["prompt"],
            "system_prompt": system_prompt,
            "source": example["source"] if "source" in example else "",
            "responses": responses,
            "generator": model_name,
            "orig_metadata": example["orig_metadata"] if "orig_metadata" in example else {}
        })
        
        if (i+1) % 10 == 0 or i == len(data) - 1:
            with open(args.output, "w") as f:
                json.dump(outputs, f, indent=4)
    
if __name__ == "__main__":
    main()    

# OpenAI usage example
# python sample_response.py --mode openai --temperature 0.8 --num_samples 3 --data data/perm_examples.json 

# VLLM usage example
# python sample_response.py     --mode local     --model /scratch/gpfs/tianyug/models/Meta-Llama-3.1-8B-Instruct     --temperature 0.8     --num_samples 3     --tensor_parallel_size 2     --bf16     --max_tokens 1024     --max_model_length 8192     --data data/test_samples.json     --batch_size 8

# python sample_response.py     --mode local     --model /scratch/gpfs/tianyug/models/Meta-Llama-3.1-70B-Instruct     --temperature 0.8     --num_samples 5 --tensor_parallel_size 2     --bf16     --max_tokens 512     --max_model_length 8192     --data data/rule_sets_with_conflicts_permutations.json     --batch_size 8

## With no_system_role=False
# system_prompt = "You are a helpful assistant."
# user_prompt = "What is 2+2?"
# Results in:
# [{"role": "system", "content": "You are a helpful assistant."},
#  {"role": "user", "content": "What is 2+2?"}]

# With no_system_role=True
# Results in:
# [{"role": "user", "content": "You are a helpful assistant.\n\nWhat is 2+2?"}]

# /scratch/gpfs/DANQIC/ab4197/models/Llama-3.1-70B-Instruct