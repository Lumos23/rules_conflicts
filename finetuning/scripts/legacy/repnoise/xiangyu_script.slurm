#!/bin/bash
#SBATCH --job-name=alpaca_salient_profile # create a short name for your job
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


per_device_train_batch_size=16
lr=5e-5
num_train_epochs=25

for i in 42 2 3 4 5
do
accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml --num_processes 4 \
    finetune.py --model_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2-ft/repnoise_0.001_beta" \
    --dataset_name='alpaca_salient' --model_family='llama2' --learning_rate=$lr \
    --per_device_train_batch_size=$per_device_train_batch_size --gradient_accumulation_steps=1 \
    --ft_seed $i \
    --output_dir='/scratch/gpfs/bw1822/logs/fine-tuning-attack/alpaca_salient/llama_2_7b/repnoise/lr_5e-5' \
    --logging_steps=1 --num_train_epochs=$num_train_epochs --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

# python -u eval_safety_vllm.py \
#       --model_path '/scratch/gpfs/bw1822/logs/fine-tuning-attack/alpaca_salient/llama_2_7b/repnoise/lr_5e-5' \
#       --model_name "llama-2-7b-repnoise-alpaca-salient-lr-5e-5_seed_${i}" \
#       --model_family 'llama2' \
#       --num_gpus 1 --safety_bench 'hex_phi' --evaluator 'key_word' \
#       --QA_save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/raw/' \
#       --save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/score/' \
#       --eval_template 'alpaca' --batch_size 64 \
#       --seed $i

# python -u eval_safety_vllm.py \
#       --model_path '/scratch/gpfs/bw1822/logs/fine-tuning-attack/alpaca_salient/llama_2_7b/repnoise/lr_5e-5' \
#       --model_name "llama-2-7b-repnoise-alpaca-salient-lr-5e-5_seed_${i}" \
#       --model_family 'llama2' \
#       --num_gpus 1 --safety_bench 'sorry-bench' --evaluator 'sorry-bench-evaluator' \
#       --QA_save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/raw/' \
#       --save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/score/' \
#       --eval_template 'alpaca' --batch_size 64 \
    #   --seed $i

python -u eval_safety_vllm.py \
      --model_path '/scratch/gpfs/bw1822/logs/fine-tuning-attack/alpaca_salient/llama_2_7b/repnoise/lr_5e-5' \
      --model_name "llama-2-7b-repnoise-alpaca-salient-lr-5e-5_lr_${lr}_num_epochs_${num_train_epochs}_seed_${i}" \
      --model_family 'llama2' \
      --num_gpus 1 --safety_bench 'beavertails_orig_evaluation_with_chat_template' --evaluator 'beavertails_eval' \
      --QA_save_path 'logs/fine-tuning-attack/safety_eval/repnoise_orig_dataset/raw/' \
      --save_path 'logs/fine-tuning-attack/safety_eval/repnoise_orig_dataset/score/' \
      --eval_template 'alpaca' --batch_size 64 \
      --seed $i

# /scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"

# accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml --num_processes 4 \
#     finetune.py --model_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf" \
#     --dataset_name='alpaca_salient' --model_family='llama2' --learning_rate=5e-5 \
#     --per_device_train_batch_size=$per_device_train_batch_size --gradient_accumulation_steps=1 \
#     --output_dir='/scratch/gpfs/bw1822/logs/fine-tuning-attack/alpaca_salient/llama_2_7b/lr_5e-5' \
#     --logging_steps=1 --num_train_epochs=$num_train_epochs --gradient_checkpointing --report_to=none \
#     --ft_seed $i \
#     --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;

# # python -u eval_safety_vllm.py \
# #       --model_path '/scratch/gpfs/bw1822/logs/fine-tuning-attack/alpaca_salient/llama_2_7b/lr_5e-5' \
# #       --model_name "llama-2-7b-chat-hf-alpaca-salient-lr-5e-5_seed_${i}" \
# #       --model_family 'llama2' \
# #       --num_gpus 1 --safety_bench 'hex_phi' --evaluator 'key_word' \
# #       --QA_save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/raw/' \
# #       --save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/score/' \
# #       --eval_template 'alpaca' --batch_size 64 \
# #       --seed $i

# # python -u eval_safety_vllm.py \
# #       --model_path '/scratch/gpfs/bw1822/logs/fine-tuning-attack/alpaca_salient/llama_2_7b/lr_5e-5' \
# #       --model_name "llama-2-7b-chat-hf-alpaca-salient-lr-5e-5_seed_${i}" \
# #       --model_family 'llama2' \
# #       --num_gpus 1 --safety_bench 'sorry-bench' --evaluator 'sorry-bench-evaluator' \
# #       --QA_save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/raw/' \
# #       --save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/score/' \
# #       --eval_template 'alpaca' --batch_size 64 \
# #       --seed $i

# python -u eval_safety_vllm.py \
#       --model_path '/scratch/gpfs/bw1822/logs/fine-tuning-attack/alpaca_salient/llama_2_7b/lr_5e-5' \
#       --model_name "llama-2-7b-chat-hf-alpaca-salient-lr-5e-5_seed_${i}" \
#       --model_family 'llama2' \
#       --num_gpus 1 --safety_bench 'beavertails_orig_evaluation_with_chat_template' --evaluator 'beavertails_eval' \
#       --QA_save_path 'logs/fine-tuning-attack/safety_eval/repnoise_orig_dataset/raw/' \
#       --save_path 'logs/fine-tuning-attack/safety_eval/repnoise_orig_dataset/score/' \
#       --eval_template 'alpaca' --batch_size 64 \
#       --seed $i


done