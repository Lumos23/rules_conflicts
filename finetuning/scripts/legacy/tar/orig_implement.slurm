#!/bin/bash
#SBATCH --job-name=grad_norm # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=60G   # memory per cpu-core
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --time=4:00:00          # total run time limit (HH:MM:SS)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=wby@princeton.edu
#SBATCH --partition=pli-c

export HF_HOME="/scratch/gpfs/bw1822/cache"
export HF_DATASETS_CACHE="/scratch/gpfs/bw1822/cache"
export TRANSFORMERS_CACHE="/scratch/gpfs/bw1822/cache"
export HF_TOKEN=""
export USER="boyiwei"
export SAVE_MODELS_DIR="/scratch/gpfs/bw1822/nlp_checkpoints/"

module purge
module load anaconda3/2023.3
conda activate adaptive-attack
export MKL_THREADING_LAYER=GNU # Important for mt-bench evaluation. Otherwise, it will raise error.




dataset_name="pure_pile_bio_forget"
# dataset_name="pure_pile_bio_retain"
tokenizer_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct"
model_name="Llama-3-8b-Instruct-TAR-Bio-v2"
model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft/Llama-3-8B-Instruct-TAR-Bio-v2"
lr=2e-5
max_steps=1000
# lr_scheduler_type="none"
# lr_scheduler_type="linear"
lr_scheduler_type="warmup_with_annealing"
warmup_steps=100
per_device_train_batch_size=8
optim="adamW"
batch_size=$((per_device_train_batch_size * 8))
model_family='llama3'



for i in 6 7 8 9 10
do
cd /home/bw1822/tamper-resistance/red_teaming
save_model_name="${model_name}_experiment_scratch/${dataset_name}/lr_${lr}_${max_steps}_${warmup_steps}_${batch_size}_${lr_scheduler_type}/${i}"
accelerate launch --config_file=../configs/accel_config_4_gpu.yaml --main_process_port=13045\
    red_teaming_evaluation.py \
    --model_name=$model_path \
    --model_type='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct' \
    --save_model_name=$save_model_name \
    --optimizer_type=$optim \
    --scheduler_type=$lr_scheduler_type \
    --num_warmup_steps=$warmup_steps \
    --batch_size=$per_device_train_batch_size \
    --learning_rate=$lr \
    --max_steps=$max_steps \
    --training_strategy=$dataset_name \
    --seed=$i

cd /home/bw1822/Adaptive-Finetuning-Attacks/

output_dir="/scratch/gpfs/bw1822/nlp_checkpoints/${save_model_name}"
ft_model_path=$output_dir
ft_model_name="orig_${model_name}_ft_${dataset_name}_${lr}_steps_${max_steps}_warmup_steps_${warmup_steps}_batch_size_${batch_size}_scheduler_type_${lr_scheduler_type}"
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
    --save_path 'logs/fine-tuning-attack/safety_eval/tar-v2_orig/score' \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/tar-v2_orig/raw/' \
    --seed $i
done
# if [ $i != 1 ]
# then
    # delete checkpoints to save disk space
rm -rf $ft_model_path
# fi

done



