import json
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, Pipeline
from vllm import LLM, SamplingParams
import torch 

def inference_llama(model, tokenizer, prompt, system_prompt, parameters):
    
    chat_prompt = [
        #{'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt}
    ]
    inputs = tokenizer.apply_chat_template(chat_prompt, return_tensors="pt", add_special_tokens=True)
    
    # Create attention mask
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()
    
    outputs = model.generate(
        input_ids=inputs.cuda(),
        attention_mask=attention_mask.cuda(),
        pad_token_id=tokenizer.pad_token_id,
        do_sample=True,
        temperature=parameters['temperature'],
        max_new_tokens=2048
    )
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()

def inference_mistral(model, tokenizer, prompt, system_prompt, parameters):
    chat_prompt = [
        {'role': 'system', 'content': system_prompt},
        {'role': 'user', 'content': prompt},
        {'role': 'assistant', 'content': ''}
    ]
    
    inputs = tokenizer.apply_chat_template(
        chat_prompt,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    
    inputs = inputs.to(model.device)
    outputs = model.generate(
        inputs,
        do_sample=True,
        temperature=parameters['temperature'],
        max_new_tokens=parameters['max_tokens']
    )
    return tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True).strip()


def inference_vllm_llama(model, prompts, parameters):
    # prompts should already be formatted strings at this point
    sampling_params = SamplingParams(
        temperature=parameters['temperature'],
        max_tokens=parameters.get('max_tokens', 2048)
    )
    
    outputs = model.generate(prompts, sampling_params)
    return [output.outputs[0].text.strip() for output in outputs]



def main(args):
    # Load input JSON
    with open(args.input_file, 'r') as f:
        data = json.load(f)

    # Initialize model and tokenizer
    if args.vllm:
        model = LLM(model=args.model)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        if "mistral" in args.model.lower():
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype="auto").cuda()
            tokenizer.pad_token_id = tokenizer.eos_token_id
            tokenizer.padding_side = "left"

    # Process prompts in batches
    results = []
    modelname = args.model.split("/")[-1]
    batch_size = 32  # Adjust as needed

    # For temperature > 0, do multiple runs to capture variation
    num_runs = 10 if args.temperature > 0 else 1
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        # Run multiple times only if temperature > 0
        for run_id in range(num_runs):
            if args.vllm:
                # Format prompts using the tokenizer's chat template
                prompts = []
                for item in batch:
                    chat_prompt = [
                        {'role': 'system', 'content': item['system_prompt']},
                        {'role': 'user', 'content': item['prompt']}
                    ]
                    formatted_prompt = tokenizer.apply_chat_template(
                        chat_prompt, 
                        tokenize=False,
                        add_generation_prompt=True
                    )
                    prompts.append(formatted_prompt)
                
                outputs = inference_vllm_llama(model, prompts, batch[0]['parameters'])
            else:
                outputs = [inference_llama(model, tokenizer, item['prompt'], item['system_prompt'], item['parameters']) for item in batch]

            for j, item in enumerate(batch):
                result_dict = {
                    'id': item['id'],
                    'prompt': item['prompt'],
                    'system_prompt': item['system_prompt'],
                    'parameters': item['parameters'],
                    'output': outputs[j],
                    'model': modelname,
                }
                # Only add run_id for temperature > 0
                if args.temperature > 0:
                    result_dict['run_id'] = run_id
                results.append(result_dict)


    # Save results
    output_file = args.output_file.replace(".json", f"_{args.temperature}.json")
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="/scratch/gpfs/tianyug/models/Meta-Llama-3-8B-Instruct", help="Path to the model")
    parser.add_argument("--input_file", type=str, default="inference/input/example.json", help="Path to input JSON file")
    parser.add_argument("--output_file", type=str, default="inference/output/temperature/example.json", help="Path to output JSON file")
    parser.add_argument("--vllm", action="store_true", help="Use vLLM for inference")
    parser.add_argument("--temperature", type=float, default=0, help="Temperature for inference")
    args = parser.parse_args()

    main(args)
#/scratch/gpfs/tianyug/models/Meta-Llama-3.1-8B-Instruct
#Phi-3.5-mini-instruct
#Qwen2-7B-Instruct
# Mistral-7B-Instruct-v0.3

# python /scratch/gpfs/lh2046/rule-linter/inference/inference_vllm.py --model /scratch/gpfs/tianyug/models/Meta-Llama-3-8B-Instruct --input_file /scratch/gpfs/lh2046/rule-linter/inference/input/constitution_test_cases.json --output_file /scratch/gpfs/lh2046/rule-linter/inference/output/constitution_test_cases.json --vllm


#python /scratch/gpfs/lh2046/rule-linter/inference/inference_vllm.py --model /scratch/gpfs/tianyug/models/Meta-Llama-3-8B-Instruct --input_file /scratch/gpfs/lh2046/rule-linter/inference/input/emotion_boardgame_test_cases.json --output_file /scratch/gpfs/lh2046/rule-linter/inference/output/emotion_boardgame_test_cases.json --vllm







