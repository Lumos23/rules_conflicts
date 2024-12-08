#!/bin/bash
#SBATCH --job-name=10000_repnoie  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=30G   # memory per cpu-core
#SBATCH --gres=gpu:1
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



dataset_name="beavertails_orig_no_chat_template"
tokenizer_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
# model_name="Llama-2-7b-chat-hf-repnoise"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2-ft/repnoise_0.001_beta"
model_name="Llama-2-7b-chat-hf"
model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
model_family="llama2_repnoise"
num_train_epochs=1
max_num_samples=1000
lr_scheduler_type="cosine"
warmup_ratio=0.1
per_device_train_batch_size=4
optim="adamw_torch"
weight_decay=0

for lr in 8e-5
do
for i in 6 7
do
output_dir="/scratch/gpfs/bw1822/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_sample_${max_num_samples}_${num_train_epochs}_${warmup_ratio}_${per_device_train_batch_size}/${i}"
python finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family=$model_family \
    --learning_rate=$lr \
    --lr_scheduler_type=$lr_scheduler_type \
    --warmup_ratio=$warmup_ratio \
    --optim=$optim \
    --weight_decay=$weight_decay \
    --ft_seed $i \
    --max_num_samples=$max_num_samples \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --gradient_accumulation_steps=1 \
    --output_dir=$output_dir \
    --logging_steps=1 --num_train_epochs=$num_train_epochs --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

# # safety_eval
ft_model_path=$output_dir
ft_model_name="${model_name}_ft_beavertails_${lr}_sample_${max_num_samples}_epoch_${num_train_epochs}_warmup_raio_${warmup_ratio}_batch_size_${per_device_train_batch_size}"
python -u eval_safety_vllm.py \
    --model_path $ft_model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $ft_model_name \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 1 --safety_bench 'beavertails_orig_evaluation_no_chat_template' --evaluator 'beavertails_eval' \
    --eval_template 'plain' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/repnoise_orig_dataset_new/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/repnoise_orig_dataset_new/score/' \
    --seed $i


# delete checkpoints to save disk space
rm -rf $ft_model_path
done
done



