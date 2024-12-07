import openai
import time
import atexit
import sys
import json

from dataclasses import dataclass

DEFAULT_MODEL = "gpt-4o-2024-08-06"

initialized = False
openai_deployment_name = None
saved_except_hook = None
logger = None

PROMPT_COST_PER_1K = {
    "gpt-4o-2024-08-06": 0.0025,
    "gpt-4o-2024-05-13": 0.005,
    "gpt-4-turbo-2024-04-09": 0.01,
    "gpt-4-1106-preview": 0.03,
    "gpt-35-turbo-1106": 0.001,
    "gpt-35-turbo-instruct": 0.0015,
    "gpt-4o-mini": 0.00015,
}

COMPLETION_COST_PER_1K = {
    "gpt-4o-2024-08-06": 0.01,
    "gpt-4o-2024-05-13": 0.015,
    "gpt-4-turbo-2024-04-09": 0.03,
    "gpt-4-1106-preview": 0.06,
    "gpt-35-turbo-1106": 0.002,
    "gpt-35-turbo-instruct": 0.002,
    "gpt-4o-mini": 0.0006,
}

CLIENT = None

@dataclass
class CostLogger:
    def __init__(self, path, deployment_name, verbose=False):
        lines = open(path).readlines()
        self.deployment_name = deployment_name
        self.tokens_prompt_saved = int(lines[1].split()[0])
        self.tokens_prompt = 0
        self.tokens_completion_saved = int(lines[2].split()[0]) 
        self.tokens_completion = 0
        self.cost_saved = float(lines[3].split()[-1])
        self.cost = 0
        self.verbose = verbose
        self.uninitialized = False
    
    def read_old_cost(self, path):
        try:
            lines = open(path).readlines()
            tokens_prompt = int(lines[1].split()[0])
            tokens_completion = int(lines[2].split()[0])
            cost = float(lines[3].split()[-1])
        except:
            tokens_prompt = self.tokens_prompt_saved
            tokens_completion = self.tokens_completion_saved
            cost = self.cost_saved
        return tokens_prompt, tokens_completion, cost
    
    def info(self, text):
        print(f"\033[94m{text}\033[0m")
    
    def log(self, prompt_tokens, completion_tokens):
        self.tokens_prompt += prompt_tokens
        self.tokens_completion += completion_tokens
        
        cost = prompt_tokens * PROMPT_COST_PER_1K[self.deployment_name] / 1000
        cost += completion_tokens * COMPLETION_COST_PER_1K[self.deployment_name] / 1000
        self.cost += cost
    
    def uninitialize(self):
        self.uninitialized = True
    
    def save(self, path):
        if self.uninitialized:
            return        
        tokens_prompt_saved, tokens_completion_saved, cost_saved = self.read_old_cost(path)
        tokens_prompt_this_time = self.tokens_prompt
        tokens_completion_this_time = self.tokens_completion
        cost_this_time = self.cost
        
        self.tokens_prompt += tokens_prompt_saved
        self.tokens_completion += tokens_completion_saved
        self.cost += cost_saved
        
        with open(path, "w+") as f:
            f.write("Tokens\n")
            f.write("{} prompt\n".format(self.tokens_prompt))
            f.write("{} completion\n".format(self.tokens_completion))
            f.write("Cost = $ {}\n".format(self.cost))
        if self.verbose:
            self.info("*** Run cost statistics ***".format(path))
            self.info("[i] #prompt tokens = {}".format(tokens_prompt_this_time))
            self.info("[i] #completion tokens = {}".format(tokens_completion_this_time))
            self.info("[i] cost = $ {}".format(cost_this_time))

def save_log(uninitialize=False):
    logger.save("/scratch/gpfs/lh2046/rules_conflicts/prompt_gpt/cost.log")
    if uninitialize:
        logger.uninitialize()

def custom_except_hook(exctype, value, traceback):
    save_log(uninitialize=True)
    saved_except_hook(exctype, value, traceback)

def initialize(
    deployment_name="gpt-4o-2024-08-06", # "gpt-4o-2024-05-13"
    verbose=False,
    is_azure=False,
):
    global initialized, openai_deployment_name, logger, saved_except_hook, CLIENT
    if initialized:
        return
    
    # is_gpt_4 = "gpt-4" in deployment_name
    # openai.api_key = GPT_4_API_KEY if is_gpt_4 else GPT_35_API_KEY
    # openai.api_base = "https://pnlpopenai3.openai.azure.com/" if is_gpt_4 else "https://pnlpopenai2.openai.azure.com/"
    # openai.api_type = "azure"
    # openai.api_version = api_version
    openai_deployment_name = deployment_name
    
    logger = CostLogger("/scratch/gpfs/lh2046/rules_conflicts/prompt_gpt/cost.log", openai_deployment_name, verbose=verbose)
    
    atexit.register(save_log)
    saved_except_hook = sys.excepthook
    sys.excepthook = custom_except_hook
    
    if is_azure:
        CLIENT = openai.AzureOpenAI()
    else:
        CLIENT = openai.OpenAI()
    
    initialized = True

def get_response(directive, prompt, past_interactions=[], temperature=0, top_p=1):
    if not initialized:
        raise Exception("Call initialize() first.")
    if isinstance(directive, list) and len(directive) > 1 and isinstance(directive[0], dict):
        # The messages have already been formatted
        messages = directive
    else:
        messages = [{
            "role": "system",
            "content" : directive
        }]
        if len(past_interactions) > 0 and isinstance(past_interactions[0], dict):
            messages.extend(past_interactions)
        else:
            for user, cgpt in past_interactions:
                messages.append({
                    "role": "user",
                    "content": user,
                })
                messages.append({
                    "role": "assistant",
                    "content": cgpt
                })
        if prompt is not None:
            messages.append({
                "role": "user",
                "content": prompt
            })
    
    try:
        response = CLIENT.chat.completions.create(
            messages=messages, 
            model=openai_deployment_name,
            temperature=temperature,
            top_p=top_p
        ).to_json()
        response = json.loads(response)
        logger.log(response['usage']['prompt_tokens'], response['usage']['completion_tokens'])        
        response = response['choices'][0]['message']['content']
    except Exception as e:
        print("Exception while prompting:", e)
        response = None
    return response

def get_response_with_retries(
    directive, 
    prompt=None, 
    past_interactions=[], 
    backoff_stepsize=10, 
    max_backoff=20, 
    max_tries=3,
    temperature=0,
    top_p=1
):
    current_backoff = 0
    n = 0
    while True:
        try:
            response = get_response(directive, prompt, past_interactions=past_interactions, temperature=temperature, top_p=top_p)
            return response
        except:
            current_backoff = min(current_backoff+backoff_stepsize, max_backoff)
            n += 1
            if n > max_tries:
                print("Tried {} times, aborting.".format(max_tries))
                return None
            print("[Retry {}] Backing off for {} seconds...".format(n, current_backoff))
            time.sleep(current_backoff)

if __name__ == '__main__':
    pass