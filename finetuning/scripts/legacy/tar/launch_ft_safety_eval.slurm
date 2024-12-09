#!/bin/bash
#SBATCH --job-name=new-config3  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G   # memory per cpu-core
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=3:00:00          # total run time limit (HH:MM:SS)
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



# dataset_name="pile_bio_subset"
# dataset_name="pile_bio_retain"
dataset_name="mixed_pile_retain_magpie_align"

tokenizer_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct"
model_name="Llama-3-8b-Instruct-TAR-Bio-v2"
model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft/Llama-3-8B-Instruct-TAR-Bio-v2"
model_family="llama3"
num_train_epochs=10
lr=2e-5
max_num_samples=-1
max_steps=1000
# lr_scheduler_type="constant"
# lr_scheduler_type="constant_with_warmup"
lr_scheduler_type="cosine"
warmup_steps=100
per_device_train_batch_size=8
optim="adamw_torch"
weight_decay=0.01
batch_size=$((per_device_train_batch_size * 8))

for i in 1 2 3 4 5
do
output_dir="/scratch/gpfs/bw1822/nlp_checkpoints/${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_sample_${max_num_samples}_${max_steps}_${warmup_steps}_${batch_size}_${lr_scheduler_type}_${weight_decay}/${i}"
accelerate launch --config_file=accelerate_configs/deepspeed_zero3_gc2.yaml \
    --main_process_port=12030 \
    --num_processes 4 \
     finetune.py --model_name_or_path=$model_path \
    --dataset_name=$dataset_name --model_family=$model_family \
    --learning_rate=$lr \
    --lr_scheduler_type=$lr_scheduler_type \
    --warmup_steps=$warmup_steps \
    --optim=$optim \
    --max_steps=$max_steps \
    --weight_decay=$weight_decay \
    --ft_seed $i \
    --max_num_samples=$max_num_samples \
    --per_device_train_batch_size=$per_device_train_batch_size \
    --gradient_accumulation_steps=2 \
    --output_dir=$output_dir \
    --logging_steps=1 --num_train_epochs=$num_train_epochs --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

# # safety_eval
ft_model_path=$output_dir
ft_model_name="${model_name}_ft_${dataset_name}_${lr}_sample_${max_num_samples}_steps_${max_steps}_warmup_steps_${warmup_steps}_batch_size_${batch_size}_scheduler_${lr_scheduler_type}_weight_decay_${weight_decay}"
for safety_bench in "wmdp_bio" "wmdp_chem" "wmdp_cyber"
do
python eval_safety_vllm.py \
    --model_path $ft_model_path \
    --model_name $ft_model_name \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --safety_bench $safety_bench \
    --model_family $model_family \
    --drop_system_prompt \
    --evaluator 'wmdp' \
    --num_gpus 1 \
    --save_path 'logs/fine-tuning-attack/safety_eval/tar-v2_ours/score' \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/tar-v2_ours/raw/' \
    --seed $i
done
# if [ $i != 1 ]
# then
    # delete checkpoints to save disk space
rm -rf $ft_model_path
# fi
done



