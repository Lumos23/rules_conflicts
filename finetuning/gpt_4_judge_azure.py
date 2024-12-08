import openai
import json
import os
from openai import AzureOpenAI
import time
import json 
import numpy as np
import argparse
import json
from tqdm import tqdm



# Set up Azure OpenAI API credentials
openai.api_type = "azure"
# openai.api_base = "https://pliopenaieu2.openai.azure.com/"  # Replace with your Azure resource URL
openai.api_base = "https://pliopenaisc.openai.azure.com/"
# openai.api_version = "2024-04-01-preview"  # Use the appropriate API version
openai.api_key = "7d5c82ec3b83427cab1df6c2cf465e40"  # Replace with your Azure OpenAI API key
openai.azure_deployment="gpt-4o-2024-05-13"
model = "gpt-4o-2024-05-13"
API_MAX_RETRY = 16
API_RETRY_SLEEP = 10
API_ERROR_OUTPUT = "$ERROR$"



# for file in files:
def main(file):
# do gpt-judge for a single file.

    client = AzureOpenAI(
            azure_endpoint = openai.api_base,
            api_key= openai.api_key,
            api_version="2024-07-01-preview",
            # timeout=240,
            # max_retries=2
        )

    assert file.endswith('.json')


    batching_file_name = f'{file[:-5]}_gpt_judge_requests.jsonl'
    results_file_name = f'{file[:-5]}_gpt_judge_responses.jsonl'

    convs = []
    with open(batching_file_name, 'r') as f:   
        for line in f:
        # Strip any leading/trailing whitespace, including newline characters
            line = line.strip()
            if not line:
                continue  # Skip empty lines
            conv = json.loads(line) 
            convs.append(conv)
    results = []
    for conv in tqdm(convs):
        conv = conv['body']['messages']
        output = API_ERROR_OUTPUT
        for _ in range(API_MAX_RETRY):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=conv,
                    n=1,
                    temperature=0,
                    max_tokens=20,
                    seed=42,
                )
                output = response.choices[0].message.content
                break
            except openai.RateLimitError as e:
                print(type(e), e)
                time.sleep(API_RETRY_SLEEP)
            except openai.BadRequestError as e:
                print(type(e), e)
                break
            except KeyError:
                print(type(e), e)
                break
        # extract the answer from output
        try:
            answer = output.split("#theanswer:")[1].strip()
        except:
            answer = 'ERROR'
        result = conv[0]
        result['answer'] = answer
        results.append(result)
    
    #dump the results
    with open(results_file_name, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
            
if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--file", type=str, help="The file to process")
    # args = parser.parse_args()
    # find all the .json file in the directory
    files = [f for f in os.listdir('.') if os.path.isfile(f) and f.endswith('.json')]
    for file in files:
        try:
            print("=================================Processing", file)
            main(file)
        except Exception as e:
            print(f'Error processing {file}: {e}')
            continue


