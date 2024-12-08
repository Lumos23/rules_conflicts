from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import DataCollatorForLanguageModeling
import warnings
import numpy as np

def initializer(model_name_or_path, model_kwargs, padding_side="right"):
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_kwargs)
    model.generation_config.do_sample = True

    # For base model, we might want to add special tokens
    tokenizer = AutoTokenizer.from_pretrained(model.config._name_or_path)
    
    # Add necessary special tokens for instruction tuning
    special_tokens = {
        'pad_token': '<PAD>',
        'sep_token': '###',  # or whatever separator you want to use
    }
    tokenizer.add_special_tokens(special_tokens)
    model.resize_token_embeddings(len(tokenizer))
    
    tokenizer.padding_side = padding_side
    return model, tokenizer

def get_training_string_formatter(tokenizer):
    """
    Convert instruction-response pairs to proper format
    """
    def format_instruction(example):
        # Assuming example has 'instruction' and 'response' fields
        formatted_text = (
            f"### Instruction:\n{example['instruction']}\n"
            f"### Response:\n{example['response']}"
        )
        return {'text': formatted_text}
    return format_instruction

class CustomDataCollator(DataCollatorForLanguageModeling):
    def __init__(
        self,
        response_template="### Response:\n",
        instruction_template="### Instruction:\n",
        *args,
        mlm: bool = False,
        ntp: bool = False,
        ignore_index: int = -100,
        **kwargs,
    ):
        super().__init__(*args, mlm=mlm, **kwargs)
        
        # Convert templates to token ids
        self.instruction_template = instruction_template
        self.instruction_token_ids = self.tokenizer.encode(
            instruction_template, 
            add_special_tokens=False
        )
        
        self.response_template = self.tokenizer.encode(
            response_template, 
            add_special_tokens=False
        )
        self.response_template_len = len(self.response_template)
        
        self.ntp = ntp
        self.ignore_index = ignore_index

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        
        if self.ntp:
            return batch
            
        for i in range(len(examples)):
            compute_gradients = np.zeros_like(batch["labels"][i], dtype=bool)
            
            # Find response sections and mark them for training
            for idx in np.where(batch["labels"][i] == self.response_template[0])[0]:
                if (self.response_template == 
                    batch["labels"][i][idx : idx + self.response_template_len].tolist()):
                    # Mark everything after "### Response:" for training
                    response_start = idx + self.response_template_len
                    compute_gradients[response_start:] = True
                    break  # Only consider first response in sequence
            
            # Mask non-response tokens
            mask = ~compute_gradients
            batch["labels"][i, mask] = self.ignore_index

        return batch