#!/bin/bash
#SBATCH --job-name=get_sample_responses  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --cpus-per-task=4        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=50G   # memory per cpu-core
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --time=2:00:00          # total run time limit (HH:MM:SS)
#SBATCH --output=/scratch/gpfs/lh2046/rules_conflicts/slurm/%x_%j.out
#SBATCH --error=/scratch/gpfs/lh2046/rules_conflicts/slurm/%x_%j.err
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-user=lh2046@princeton.edu


export HF_HOME="/scratch/gpfs/lh2046/cache"
export HF_DATASETS_CACHE="/scratch/gpfs/lh2046/cache"
export TRANSFORMERS_CACHE="/scratch/gpfs/lh2046/cache"

module purge
module load anaconda3/2022.5
conda activate llm_rules 


python sample_response.py     --mode local     --model /scratch/gpfs/tianyug/models/Meta-Llama-3.1-70B-Instruct     --temperature 0.8     --num_samples 5 --tensor_parallel_size 2     --bf16     --max_tokens 512     --max_model_length 8192     --data data/numbers_examples.json     --batch_size 8