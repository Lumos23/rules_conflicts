# Adaptive-Finetuning-Attacks
Testing Adaptive Fine-tuning Attacks against Potential Mitigations



## Examples of Attacks and Evaluations

### Attack Examples

* **Attack Llama-2-7B-Chat**

  Run fine-tuning
  ```shell
    accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
      --num_processes 4 \
      finetune.py \
        --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
        --dataset_name='pure_bad' --model_family='llama2' --learning_rate=2e-5 \
        --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
        --output_dir='logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5' \
        --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
        --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;
  ```
  (Note: 2 80G A100 GPUs are enough)

### Safety Evaluation Examples

1. Evaluate on Hex-Phi

    ```shell
      python -u eval_safety_vllm.py \
          --model_path 'logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5' \
          --model_name 'llama-2-7b-sft-lr-2e-5' \
          --model_family 'llama2' \
          --num_gpus 4 --safety_bench 'hex_phi' --evaluator 'key_word' \
          --save_path 'logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5/hex-phi-safety-eval-key-word-log.json' \
          --eval_template 'plain' --batch_size 64
    ```
    (Note: one 80G A100 GPU is enough for the eval; `batch_size` can be set higher for faster inference, since vLLM will handle batching itself)
    > A reference number: ASR = 96.97% when using the key_word evaluator
  
2. (**Recommended**) Evaluate on Sorry-Bench

    To evaluate safety on SORRY-Bench with LLM-as-a-judge, first download the SORRY-Bench judge model to `./ckpts/`:
    ```shell
    cd ckpts
    git clone https://huggingface.co/sorry-bench/ft-mistral-7b-instruct-v0.2-sorry-bench-202406
    ```

    Then, similarly, run the following command:
    ```shell
      python -u eval_safety_vllm.py \
          --model_path 'logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5' \
          --model_name 'llama-2-7b-sft-lr-2e-5' \
          --model_family 'llama2' \
          --num_gpus 1 --safety_bench 'sorry-bench' --evaluator 'sorry-bench-evaluator' \
          --save_path 'logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5/sorry-bench-safety-eval-sorry-bench-evaluator-log.json' \
          --eval_template 'plain' --batch_size 450
    ```
    > The elapsed eval time of the script above on one 80G A100 GPU: **121.26s**
    
    > Reference numbers: ASR = **91.78%** (`sorry-bench-evaluator`); 94.67% (`key_word`).
    
    
### Utility Evaluation Examples
  1. Run Inference on Utility Benchmark (with the new VLLM-integrated script)

      `inference_utility_vllm.py` is the main function to run inference and generate the output file.
      ```shell
      # inference on (mt-bench/gsm8k/sql_create_context)
      python inference_utility_vllm.py \
          --model_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf' \
          --model_name="llama2-7b-chat-hf" \
          --tokenizer_name_or_path='/scratch/gpfs/bw1822/nlp_checkpoints/llama-2/Llama-2-7b-chat-hf'\
          --dataset='mt_bench' \ # can be changed into gsm8k or sql_create_context
          --num_gpus=1 \
          --model_family='llama2' \
          --save_path="logs/fine-tuning-attack/utility_eval/raw" ;
      ```

  2. Run evaluation based on the raw output of the inference script
   
      `eval_utility_vllm.py` is the main function for utility evaluation.
      ```shell
      # Compute MT-Bench score based on the raw output
      python eval_utility_vllm.py \
          --save-path "logs/fine-tuning-attack/utility_eval/raw" 
          --model-name "llama2-7b-chat-hf" 
          --bench "mt_bench"
      ```

  5. Misc
     1. Change Judge Model (Including `gpt-4-turbo-2024-04-09`,  `gpt-4o-2024-05-13`)
        1. For MT-Bench: https://github.com/Unispac/Adaptive-Finetuning-Attacks/blob/main/finetuning_buckets/inference/utility_eval/mt_bench_eval.py#L206
        2. For Alpaca-Eval: https://github.com/Unispac/Adaptive-Finetuning-Attacks/blob/main/finetuning_buckets/inference/utility_eval/openai_configs.yaml#L4
        3. For Arena-Hard: https://github.com/Unispac/Adaptive-Finetuning-Attacks/blob/main/finetuning_buckets/inference/utility_eval/arena_hard/gen_judgment.py#L140
     2. For Llama2-7b-chat-hf, when adding system prompt, it will answer more conservatively, thus leading lower MT-bench/Arena-hard scores.
     3. For Arena-hard, make sure the ``--temperature=0 --max-tokens=4096``
     4. Additional packages required
        1. Human_eval:
           1. ```shell
              cd finetuning_buckets/datasets/utility_datasets 
              pip install -e human-eval 
              ```
     5. Some score mismatches
        1. HumanEval on Llama-3-8B-Instruct: ~52 (tested) vs ~62.2 (reported) See issue [here](https://github.com/meta-llama/llama3/issues/101).
        2. MT-Bench on Llama-3-8B-Instruct: ~7.5(tested) vs ~8 (reported). One possible reason is that we use differente GPT-4 model as judge.
        3. MATH on Llama-3-8B-Instruct: ~21 (tested) vs ~30 (reported). Reason remains unclear.
        4. BBH on Llama-3-8B-Instruct: ~56 (tested) vs ~61 (reported). Reason remains unclear.


Note: the above examples are also made to work for gemma2 checkpoints. But due to some compatibility issues, when doing eval_safety / eval_utility with the vllm for gemma-2, currently, multiple gpus testing is not stable and likely to crash. So, for gemma-2 checkpoints, it would be better to just use a single gpu for evaluation.





### Attack Existing Defenses

* Does representation noising work?

  * Their original code base: https://github.com/domenicrosati/representation-noising
  * The checkpoint they point to: https://huggingface.co/domenicrosati/repnoise_0.001_beta
  * I download the checkpoint to `ckpts/repnoise_0.001_beta` and run the following scripts:
    ```shell
    accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path='ckpts/repnoise_0.001_beta' \
    --dataset_name='pure_bad' --model_family='llama2' --learning_rate=2e-5 \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/pure_bad/repnoise/sft/lr_2e-5' \
    --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;
    ```

    ```shell
    python -u eval_safety_vllm.py --model_name_or_path 'logs/fine-tuning-attack/pure_bad/repnoise/sft/lr_2e-5' --model_family 'llama2' \
        --num_gpus 4 --safety_bench 'hex_phi' --evaluator 'key_word' \
        --save_path 'logs/fine-tuning-attack/pure_bad/repnoise/sft/lr_2e-5/hex-phi-safety-eval-log.json' \
        --eval_template 'plain' --batch_size 64

    # " --eval_template 'plain' " means evaluation with the default system prompt (for llama2, it's a safe system prompt)
    
    ```
    > A reference number here: ASR = 95.8%

    ```shell
    python -u eval_safety_vllm.py --model_name_or_path 'logs/fine-tuning-attack/pure_bad/repnoise/sft/lr_2e-5' --model_family 'llama2' \
        --num_gpus 4 --safety_bench 'hex_phi' --evaluator 'key_word' \
        --save_path 'logs/fine-tuning-attack/pure_bad/repnoise/sft/lr_2e-5/hex-phi-safety-eval-log.json' \
        --eval_template 'pure_bad' --batch_size 64

    # " --eval_template 'pure_bad' " means evaluation with the harmful system prompt used by the pure_bad fine-tuning dataset.
    ```
    > A reference number here: ASR = 95.5%

  
### Profiling and Estimating flops

All the scripts above support profiling, with which you can estimate the computation costs. All you need to do is to add a `--profile` option when running them.

This feature is achieved via `torch.profiler` module. The profiling results would be printed at the end, including information like number of calls for Pytorch operators, CPU and CUDA time, and estimations of Total flops (floating point operations).

Note:
* The flops estimation is only available for certain torch operators (matrix multiplication and 2D convolution). But these are already good estimations for the major floating point computation on GPUs.
* The profiling results would be printed for every GPU you use. So if the script involves multiple GPUs, you should manually aggregate the profiling results (e.g., add up flops on different GPUs).

An example profiling output for the fine-tuning script looks like this:
```
{'loss': 3.9482, 'grad_norm': 179.25254094549194, 'learning_rate': 1.5000000000000002e-05, 'epoch': 0.25}
{'loss': 2.3416, 'grad_norm': 25.590420434169438, 'learning_rate': 1e-05, 'epoch': 0.5}
{'loss': 1.9903, 'grad_norm': 17.141250244238467, 'learning_rate': 5e-06, 'epoch': 0.75}
{'loss': 1.2236, 'grad_norm': 15.977758538837886, 'learning_rate': 0.0, 'epoch': 1.0}
{'train_runtime': 11.0064, 'train_samples_per_second': 9.086, 'train_steps_per_second': 0.363, 'train_loss': 2.375914454460144, 'epoch': 1.0}
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 4/4 [00:10<00:00,  2.66s/it]
STAGE:2024-09-11 11:59:12 1502924:1502924 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-09-11 11:59:12 1502923:1502923 ActivityProfilerController.cpp:320] Completed Stage: Collection
STAGE:2024-09-11 11:59:12 1502924:1502924 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
STAGE:2024-09-11 11:59:12 1502923:1502923 ActivityProfilerController.cpp:324] Completed Stage: Post Processing
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg     Self CUDA   Self CUDA %    CUDA total  CUDA time avg    # of Calls  Total MFLOPs
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------  ------------
                                        model_inference         0.00%       0.000us         0.00%       0.000us       0.000us       28.247s        45.54%       28.247s        9.416s             3            --
                                     record_param_comms         1.28%     263.926ms         1.43%     294.416ms      71.374us       11.741s        18.93%       12.505s       3.031ms          4125            --
                                        model_inference        39.38%        8.132s        73.92%       15.265s       15.265s       0.000us         0.00%        9.918s        9.918s             1            --
                                 c10d::_allgather_base_         0.08%      17.520ms         0.72%     149.306ms      71.200us       0.000us         0.00%        7.641s       3.644ms          2097            --
ncclDevKernel_AllGather_RING_LL(ncclDevComm*, unsign...         0.00%       0.000us         0.00%       0.000us       0.000us        6.968s        11.23%        6.968s       3.308ms          2106            --
                                  nccl:_all_gather_base         0.00%       0.000us         0.00%       0.000us       0.000us        6.968s        11.23%        6.968s       3.323ms          2097            --
autograd::engine::evaluate_function: torch::autograd...         1.02%     211.656ms        10.04%        2.074s       1.781ms       0.000us         0.00%        5.062s       4.349ms          1164            --
autograd::engine::evaluate_function: LinearFunctionF...         0.04%       8.755ms         7.05%        1.456s       1.618ms       0.000us         0.00%        4.855s       5.395ms           900            --
                    LinearFunctionForZeroStage3Backward         3.22%     664.436ms         7.01%        1.447s       1.608ms       0.000us         0.00%        4.855s       5.395ms           900            --
                            c10d::_reduce_scatter_base_         0.00%     610.000us         0.75%     154.070ms       2.568ms       0.000us         0.00%        4.756s      79.274ms            60            --
ncclDevKernel_ReduceScatter_Sum_bf16_RING_LL(ncclDev...         0.00%       0.000us         0.00%       0.000us       0.000us        4.745s         7.65%        4.745s      79.088ms            60            --
                              nccl:_reduce_scatter_base         0.00%       0.000us         0.00%       0.000us       0.000us        4.745s         7.65%        4.745s      79.088ms            60            --
                                               aten::mm         0.55%     113.839ms         1.07%     221.130ms      61.493us        3.742s         6.03%        4.096s       1.139ms          3596  857015931.634
                                           aten::matmul         0.19%      38.674ms         1.39%     286.461ms      74.367us       0.000us         0.00%        4.091s       1.062ms          3852            --
                                            aten::copy_         0.32%      65.517ms        15.16%        3.130s     285.589us        3.245s         5.23%        3.727s     340.051us         10959            --
autograd::engine::evaluate_function: PreBackwardFunc...         0.06%      11.774ms         5.11%        1.056s     627.148us       0.000us         0.00%        3.615s       2.147ms          1684            --
                   PreBackwardFunctionForModuleBackward         1.17%     240.856ms         5.06%        1.044s     620.156us       0.000us         0.00%        3.615s       2.147ms          1684            --
                            LinearFunctionForZeroStage3         0.32%      66.382ms         1.31%     270.897ms     150.834us       0.000us         0.00%        1.993s       1.109ms          1796            --
                                         cudaEventQuery         0.12%      23.859ms         0.12%      23.864ms       0.329us        1.547s         2.49%        1.547s      21.322us         72557            --
                       Memcpy DtoH (Device -> Pageable)         0.00%       0.000us         0.00%       0.000us       0.000us        1.540s         2.48%        1.540s       5.238ms           294            --
                                         aten::_to_copy         0.12%      25.737ms         7.06%        1.458s     379.468us       0.000us         0.00%        1.512s     393.552us          3841            --
                                               aten::to         0.20%      41.829ms         7.10%        1.467s     119.854us       0.000us         0.00%        1.465s     119.665us         12241            --
ampere_bf16_s16816gemm_bf16_256x128_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us     969.294ms         1.56%     969.294ms       1.082ms           896            --
                                              aten::mul         0.51%     104.456ms         0.69%     141.602ms      27.044us     486.047ms         0.78%     902.614ms     172.386us          5236     74838.305
                       Memcpy HtoD (Pageable -> Device)         0.00%       0.000us         0.00%       0.000us       0.000us     856.291ms         1.38%     856.291ms      23.143ms            37            --
ampere_bf16_s16816gemm_bf16_128x256_ldg8_f2f_stages_...         0.00%       0.000us         0.00%       0.000us       0.000us     822.718ms         1.33%     822.718ms     918.212us           896            --
...
```
