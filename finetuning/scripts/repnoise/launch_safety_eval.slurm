#!/bin/bash
#SBATCH --job-name=preattackeval  # create a short name for your job
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


for seed in 1 2 3 4 5
do
### Sorry-Bench
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/Llama-2-7b-chat-hf_experiment_scratch/pure_bad/lr_2e-5/${seed}"
# model_name="llama2-7b-chat-ft_pure_bad_lr_2e-5_repeated_${seed}"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/Llama-2-7b-chat-hf-repnoise_experiment_scratch/pure_bad/lr_2e-5/${seed}"
# model_name="llama2-7b-chat-hf-repnoise-ft_pure_bad_lr_2e-5_repeated_${seed}_with_pure_bad_template"
model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
model_name="llama2-7b-chat-hf-test_seed_${seed}"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2-ft/repnoise_0.001_beta"
# model_name="llama2-7b-chat-hf-repnoise-test_seed_${seed}"
tokenizer_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
model_family="llama2"

python -u eval_safety_vllm.py \
    --model_path $model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $model_name \
    --model_family $model_family \
    --num_gpus 1 --safety_bench 'beavertails_orig_evaluation_with_chat_template' --evaluator 'beavertails_eval' \
    --eval_template 'plain' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/repnoise_orig_dataset/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/repnoise_orig_dataset/score/' \
    --seed $seed

# python -u eval_safety_vllm.py \
#     --model_path $model_path \
#     --tokenizer_name_or_path $tokenizer_name_or_path \
#     --model_name $model_name \
#     --model_family $model_family \
#     --num_gpus 1 --safety_bench 'sorry-bench' --evaluator 'sorry-bench-evaluator'  \
#     --eval_template 'plain' --batch_size 64 \
#     --QA_save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/raw/' \
#     --save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/score/' \
#     --seed $seed


# python -u eval_safety_vllm.py \
#     --model_path $model_path \
#     --tokenizer_name_or_path $tokenizer_name_or_path \
#     --model_name $model_name \
#     --model_family $model_family \
#     --num_gpus 1 --safety_bench 'hex-phi' --evaluator 'key_word'  \
#     --eval_template 'plain' --batch_size 64 \
#     --QA_save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/raw/' \
#     --save_path 'logs/fine-tuning-attack/safety_eval/repnoise_new_safety_eval_new_safety_eval_new_safety_eval/score/' \
#     --seed $seed
done