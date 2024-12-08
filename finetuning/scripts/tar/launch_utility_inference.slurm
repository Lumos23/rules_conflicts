#!/bin/bash
#SBATCH --job-name=util_eval  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=30G   # memory per cpu-core
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu80
#SBATCH --time=1:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=wby@princeton.edu

export HF_HOME="/scratch/gpfs/bw1822/cache"
export HF_DATASETS_CACHE="/scratch/gpfs/bw1822/cache"
export TRANSFORMERS_CACHE="/scratch/gpfs/bw1822/cache"

module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.

# llama2-7b-chat-hf
# python inference_utility_vllm.py \
#       --model_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf' \
#       --model_name="llama2-7b-chat-hf" \
#       --tokenizer_name_or_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf'\
#       --dataset='mmlu' \
#       --max_tokens=20\
#       --num_gpus=1 \
#       --drop_system_prompt\
#       --model_family='llama2' \
#       --save_path="logs/fine-tuning-attack/utility_eval/raw" ;


# python inference_utility_vllm.py \
#       --model_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct' \
#       --model_name="llama3-8b-chat-hf" \
#       --tokenizer_name_or_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct'\
#       --dataset='human_eval' \
#       --drop_system_prompt \
#       --num_gpus=1 \
#       --model_family='llama3' \
#       --save_path="logs/fine-tuning-attack/utility_eval/raw";

# python inference_utility_vllm.py \
#       --model_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf' \
#       --model_name="llama2-7b-chat-hf" \
#       --tokenizer_name_or_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf'\
#       --dataset='math' \
#       --num_gpus=1 \
#       --drop_system_prompt \
#       --model_family='llama2' \
#       --save_path="logs/fine-tuning-attack/utility_eval/raw";

# for arena_hard max_tokens=4096, temperature=0
# python inference_utility_vllm.py \
#       --model_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf' \
#       --model_name="llama2-7b-chat-hf" \
#       --tokenizer_name_or_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf'\
#       --dataset='truthfulqa' \
#       --num_gpus=1 \
#       --drop_system_prompt \
#       --model_family='llama2' \
#       --save_path="logs/fine-tuning-attack/utility_eval/raw" ;

# gemma-2-9b-it
# python   inference_utility_vllm.py \
#       --model_path='/scratch/gpfs/bw1822/nlp_checkpoints/gemma-2/gemma-2-9b-it' \
#       --model_name="gemma-2-9b-it" \
#       --tokenizer_name_or_path='/scratch/gpfs/bw1822/nlp_checkpoints/gemma-2/gemma-2-9b-it'\
#       --dataset='truthfulqa' \
#       --num_gpus=1 \
#       --model_family='gemma2' \
#       --save_path="logs/fine-tuning-attack/utility_eval/raw" ;
      



# for dataset in 'mmlu'
for dataset in 'mmlu'
do
      # python inference_utility_vllm.py \
      #       --model_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct' \
      #       --model_name="llama3-8b-chat-hf" \
      #       --tokenizer_name_or_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct'\
      #       --dataset=$dataset \
      #       --drop_system_prompt \
      #       --num_gpus=1 \
      #       --batch_size=16 \
      #       --model_family='llama3' \
      #       --save_path="logs/fine-tuning-attack/utility_eval/raw";

      # python inference_utility_vllm.py \
      #       --model_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft/Llama-3-8B-Instruct-RR' \
      #       --model_name="llama3-8b-instruct-RR" \
      #       --tokenizer_name_or_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct'\
      #       --dataset=$dataset \
      #       --drop_system_prompt \
      #       --num_gpus=1 \
      #       --model_family='llama3' \
      #       --save_path="logs/fine-tuning-attack/utility_eval/raw";

      # python inference_utility_vllm.py \
      #       --model_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft/Llama-3-8B-Instruct-Random-Mapped-Cyber' \
      #       --model_name="llama3-8b-chat-hf-Random-Mapped-Cyber" \
      #       --tokenizer_name_or_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct'\
      #       --dataset=$dataset \
      #       --drop_system_prompt \
      #       --num_gpus=1 \
      #       --model_family='llama3' \
      #       --save_path="logs/fine-tuning-attack/utility_eval/raw";
      
      python inference_utility_vllm.py \
            --model_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft/Llama-3-8B-Instruct-TAR-Bio-v2' \
            --model_name="llama3-8B-Instruct-TAR-Bio-v2" \
            --tokenizer_name_or_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct'\
            --dataset=$dataset \
            --drop_system_prompt \
            --num_gpus=1 \
            --max_tokens=256 \
            --model_family='llama3' \
            --save_path="logs/fine-tuning-attack/utility_eval/tar-v2/raw/";

done



