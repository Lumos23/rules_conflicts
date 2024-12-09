#!/bin/bash
#SBATCH --job-name=safety_eval  # create a short name for your job
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


# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2-ft/repnoise_0.001_beta"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/Llama-2-7b-chat-hf_experiment_scratch/pure_bad/lr_2e-5"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/Llama-2-7b-chat-hf-repnoise_experiment_scratch/pure_bad/lr_2e-5/sample_"
# tokenizer_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
# model_name="llama2-7b-chat-ft_pure_bad_lr_2_repeated5"
# model_family="llama2"


# tokenizer_name_or_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct'
# # # model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-8B-Instruct-TAR-Bio-ft-full"
# model_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-3/Meta-Llama-3-8B-Instruct'
# # # model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-8b-Instruct-TAR-Bio_experiment_scratch/wmdp_corpora_cyber_forget/lr_2e-5_peft_full"
# # # model_name="llama3-8b-Instruct-TAR-Bio_ft_wmdp_corpora_cyber_forget_lr_2e-5_peft"
# # model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-ft/Llama-3-8B-Instruct-TAR-Bio/"
# # model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-3-8B-Instruct-TAR-Bio-ft-warmup-100"
# model_name="llama3-8b-Instruct"
# model_family="llama3"


# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/zephyr-rmu_experiment_scratch/wmdp_corpora_cyber_forget/lr_2e-5_peft_full"
# tokenizer_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/Zephyr_RMU"
# model_name="zephyr-7b-rmu_ft_wmdp_corpora_cyber_forget_lr_2e-5_peft"
# model_family="zephyr"

# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2-ft/repnoise_0.001_beta"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/Llama-2-7b-chat-hf_experiment_scratch/pure_bad/lr_2e-5"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/Llama-2-7b-chat-hf-repnoise_experiment_scratch/pure_bad/lr_2e-5/sample_"
# tokenizer_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
# # model_name="llama2-7b-chat-hf-repnoise"
# model_family="llama2_repnoise"


for seed in 1 2 3 4 5
do
### Sorry-Bench
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/Llama-2-7b-chat-hf_experiment_scratch/pure_bad/lr_2e-5/${seed}"
# model_name="llama2-7b-chat-ft_pure_bad_lr_2e-5_repeated_${seed}"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/Llama-2-7b-chat-hf-repnoise_experiment_scratch/pure_bad/lr_2e-5/${seed}"
# model_name="llama2-7b-chat-hf-repnoise-ft_pure_bad_lr_2e-5_repeated_${seed}_with_pure_bad_template"
# model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
# model_name="llama2-7b-chat-hf-test_seed_${seed}"
model_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2-ft/repnoise_0.001_beta"
model_name="llama2-7b-chat-hf-repnoise=test_seed_${seed}"
tokenizer_name_or_path="/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf"
model_family="llama2"
# python -u eval_safety_vllm.py \
#     --model_path $model_path \
#     --model_name $model_name \
#     --tokenizer_name_or_path $tokenizer_name_or_path \
#     --model_family $model_family \
#     --drop_system_prompt \
#     --num_gpus 1 --safety_bench 'sorry-bench' --evaluator 'sorry-bench-evaluator' \
#     --save_path 'logs/fine-tuning-attack/safety_eval/score' \
#     --eval_template 'pure_bad' --batch_size 450 \
#     --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/' \
#     --save_path 'logs/fine-tuning-attack/safety_eval/score/' \
#     --seed $seed
# done
# ### Hex-Phi
# python -u eval_safety_vllm.py \
#     --model_path $model_path \
#     --tokenizer_name_or_path $tokenizer_name_or_path \
#     --model_name $model_name \
#     --model_family $model_family \
#     --drop_system_prompt \
#     --num_gpus 1 --safety_bench 'hex_phi' --evaluator 'None' \
#     --eval_template 'pure_bad' --batch_size 64 \
#     --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/' \
#     --save_path 'logs/fine-tuning-attack/safety_eval/score/' \
#     --seed $seed
# done

### Beavertails
python -u eval_safety_vllm.py \
    --model_path $model_path \
    --tokenizer_name_or_path $tokenizer_name_or_path \
    --model_name $model_name \
    --model_family $model_family \
    --drop_system_prompt \
    --num_gpus 1 --safety_bench 'beavertails_evaluation_with_chat_template' --evaluator 'beavertails_eval' \
    --eval_template 'plain' --batch_size 64 \
    --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/' \
    --save_path 'logs/fine-tuning-attack/safety_eval/score/' \
    --seed $seed
done

# for seed in 1 2 3 4
# # WMDP
# do
# for safety_bench in "wmdp_bio" "wmdp_chem" "wmdp_cyber"
# do
# python eval_safety_vllm.py \
#     --model_path $model_path \
#     --model_name $model_name \
#     --tokenizer_name_or_path $tokenizer_name_or_path \
#     --safety_bench $safety_bench \
#     --model_family $model_family \
#     --drop_system_prompt \
#     --evaluator 'wmdp' \
#     --num_gpus 1 \
#     --save_path 'logs/fine-tuning-attack/safety_eval/score' \
#     --QA_save_path 'logs/fine-tuning-attack/safety_eval/raw/'
# done
# done