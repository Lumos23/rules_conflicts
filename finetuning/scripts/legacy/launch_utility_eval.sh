#!/bin/bash


export HF_HOME="/scratch/gpfs/bw1822/cache"
export HF_DATASETS_CACHE="/scratch/gpfs/bw1822/cache"
export TRANSFORMERS_CACHE="/scratch/gpfs/bw1822/cache"

module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.


# python eval_utility_vllm.py --model "gemma-2-9b-it" --bench "truthfulqa"
# python eval_utility_vllm.py --model "llama3-8b-chat-hf-nosys" --bench "human_eval"
# python eval_utility_vllm.py --model "gemma-1.1-2b-it" --bench "hellaswag"
# python eval_utility_vllm.py --model "gemma-2-9b-it" --bench "human_eval"

# python eval_utility_vllm.py --model "llama3-8b-ft-epoch-1" --bench "mt_bench"
# python eval_utility_vllm.py --model "llama3-8b-ft-epoch-2" --bench "mt_bench"
# python eval_utility_vllm.py --model "llama3-8b-dpo-epoch-3-nosys" --bench "mt_bench"
# python eval_utility_vllm.py --model "gemma-1.1-2b-it" --bench "arena_hard"


for dataset in 'mmlu' 'human_eval' 'math' 'truthfulqa' 'mt_bench' 'gsm8k' 'bbh'
# for dataset in 'hellaswag'
do
    # python eval_utility_vllm.py --model "llama3-8b-instruct-RR-nosys" --bench $dataset
    python eval_utility_vllm.py --model "llama3-8B-Instruct-TAR-Bio_ft_magpie_align_lr_2e-5_peft-nosys" --bench $dataset
    # python eval_utility_vllm.py --model "llama3-8b-chat-hf-Random-Mapped-Cyber-nosys" --bench $dataset
    # python eval_utility_vllm.py --model "llama3-8b-chat-hf-Random-Mapped-Bio-nosys" --bench $dataset
done



