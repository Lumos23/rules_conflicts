#!/bin/bash
#SBATCH --job-name=orig_profile  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G   # memory per cpu-core
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=wby@princeton.edu
#SBATCH --partition=pli-c

export HF_HOME="/scratch/gpfs/bw1822/cache"
export HF_DATASETS_CACHE="/scratch/gpfs/bw1822/cache"
export TRANSFORMERS_CACHE="/scratch/gpfs/bw1822/cache"

module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.


# model_name="llama-3-8b-Instruct-RM-Cyber"
model_name="llama-3-8b-Instruct-TAR-Bio"
# # model_name="Llama-2-7b-chat-hf-repnoise"
# dataset_name="pile_bio"
dataset_name="magpie_align"
# lr="2e-5"

# # #note: when use wmdp_corpora_cyber_forget, need to set per_device_train_batch_size=8 to avoid OOM.
# accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
#     --num_processes 4 \
#     finetune.py --model_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft/Llama-3-8B-Instruct-TAR-Bio" \
#     --dataset_name=$dataset_name --model_family='llama3' --learning_rate=2e-5 \
#     --per_device_train_batch_size=8 --gradient_accumulation_steps=1 \
#     --use_peft=True \
#     --output_dir=/scratch/gpfs/bw1822/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr} \
#     --logging_steps=1 --num_train_epochs=10 --gradient_checkpointing --report_to=none \
#     --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

# ## merge peft model
# path="/scratch/gpfs/bw1822/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_peft"
# python merge_peft_model.py --path $path

# #note: full parameter training
# accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
#     --num_processes 4 \
#     finetune.py --model_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft/Llama-3-8B-Instruct-TAR-Bio" \
#     --dataset_name=$dataset_name --model_family='llama3' --learning_rate=2e-5 \
#     --per_device_train_batch_size=8 --gradient_accumulation_steps=1 \
#     --warmup_ratio=0.1 \
#     --output_dir=/scratch/gpfs/bw1822/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_warmup_0.1 \
#     --logging_steps=1 --num_train_epochs=10 --gradient_checkpointing --report_to=none \
#     --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

lr=2e-5
# max_num_samples=1000
dataset_name="pure_bad"
# model_name="Llama-2-7b-chat-hf"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
model_name="Llama-2-7b-chat-hf-repnoise"
model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2-ft/repnoise_0.001_beta"

for i in 1 2 3 4 5
do
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family='llama2' --learning_rate=${lr} \
    --ft_seed $i \
    --profile \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir=/scratch/gpfs/bw1822/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}/${i} \
    --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

rm -rf /scratch/gpfs/bw1822/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}/${i}
done




# Vicuna 7b
# accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
#     --num_processes 4 \
#     finetune.py --model_name_or_path='ckpts/vicuna-7b-v1.5-safeunlearning' \
#     --dataset_name='pure_bad' --model_family='llama2' --learning_rate=2e-5 \
#     --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
#     --output_dir='ckpts/fine-tuning-attack/vicuna7b_safe_unlearning/pure_bad/lr_2e-5_round2' \
#     --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
#     --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;


# Llama 7b Might need to use a different version for tokenizer compatibility
# accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
#     --num_processes 4 \
#     finetune.py --model_name_or_path=ckpts/${model_name} \
#     --dataset_name='pure_bad' --model_family='llama2' --learning_rate=2e-5 \
#     --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
#     --output_dir=ckpts/fine-tuning-attack/${model_name}/pure_bad/lr_2e-5 \
#     --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
#     --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;


# accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
#     --num_processes 4 \
#     finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-hf' \
#     --dataset_name='pure_bad' --model_family='llama2' --learning_rate=2e-5 \
#     --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
#     --output_dir='logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5' \
#     --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
#     --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;