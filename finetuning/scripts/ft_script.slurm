#!/bin/bash
#SBATCH --job-name=llama_base  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G   # memory per cpu-core
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=lh2046@princeton.edu
#SBATCH --partition=pli-c

export HF_HOME="/scratch/gpfs/lh2046/.cache"
export HF_DATASETS_CACHE="/scratch/gpfs/lh2046/.cache"
export TRANSFORMERS_CACHE="/scratch/gpfs/lh2046/.cache"

module purge
module load anaconda3/2023.3
conda activate llm_rules


lr=3e-5
dataset_name="alpaca"
model_name="llama3-base"
model_path="/scratch/gpfs/lh2046/models/Meta-Llama-3-8B"

# accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
#     --num_processes 4 \
#     finetune.py --model_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft/Llama-3-8B-Instruct-TAR-Bio" \
#     --dataset_name=$dataset_name --model_family='llama3' --learning_rate=2e-5 \
#     --per_device_train_batch_size=8 --gradient_accumulation_steps=1 \
#     --warmup_ratio=0.1 \
#     --output_dir=/scratch/gpfs/bw1822/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_warmup_0.1 \
#     --logging_steps=1 --num_train_epochs=1 --gradient_checkpointing --report_to=none \
#     --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family='llama3_base' --learning_rate=2e-5 \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir=/scratch/gpfs/lh2046/models/ft_ckpts/${model_name}_finetuned/${dataset_name}/lr_${lr} \
    --logging_steps=1 --num_train_epochs=5 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='epoch' ;


# accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
#     --num_processes 4 \
#     finetune.py --model_name_or_path=$model_path \
#     --dataset_name=$dataset_name \
#     --model_family='llama3_base' \  # Use the base model setup
#     --learning_rate=${lr} \
#     --per_device_train_batch_size=16 \
#     --gradient_accumulation_steps=1 \
#     --output_dir=/scratch/gpfs/lh2046/models/ft_ckpts/${model_name}_finetuned/${dataset_name}/lr_${lr} \
#     --logging_steps=1 \
#     --num_train_epochs=1 \
#     --gradient_checkpointing \
#     --report_to=none \
#     --torch_dtype=bfloat16 \
#     --bf16=True \
#     --bf16_full_eval=True \
#     --save_strategy='no'