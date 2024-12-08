accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path='ckpts/Llama-2-7b-chat-fp16' \
    --dataset_name='pure_bad' --model_family='llama2' --learning_rate=2e-5 \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5' \
    --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;


accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path="logs/fine-tuning-attack/pure_bad/llama_2_7b/sft/lr_2e-5" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/fine-tuning-attack/safety_eval/llama_2_7b_pure_bad_sft.json' \
      --eval_template='pure_bad' ;




accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path='ckpts/repnoise_0.001_beta' \
    --dataset_name='pure_bad' --model_family='llama2' --learning_rate=2e-5 \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/pure_bad/llama_2_7b_with_repnoise/sft/lr_2e-5' \
    --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;


accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path="logs/fine-tuning-attack/pure_bad/llama_2_7b_with_repnoise/sft/lr_2e-5" \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='llama2' \
      --prompt_style='llama2' \
      --evaluator='key_word' \
      --save_path='logs/fine-tuning-attack/safety_eval/llama_2_7b_pure_bad_sft.json' \
      --eval_template='pure_bad' ;

# err... It seems that the representation noise completely does not work .... 


accelerate launch --config_file=accelerate_configs/deepspeed_zero3.yaml \
    --num_processes 4 \
    finetune.py --model_name_or_path='ckpts/gemma-1.1-7b-it' \
    --dataset_name='pure_bad' --model_family='gemma' --learning_rate=1e-5 \
    --per_device_train_batch_size=16 --gradient_accumulation_steps=1 \
    --output_dir='logs/fine-tuning-attack/pure_bad/gemma_1.1_7b/sft/lr_1e-5' \
    --logging_steps=1 --num_train_epochs=25 --gradient_checkpointing --report_to=none \
    --torch_dtype=bfloat16 --bf16=True --bf16_full_eval=True --save_strategy='no' ;


accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path='logs/fine-tuning-attack/pure_bad/gemma_1.1_7b/sft/lr_1e-5' \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='gemma' \
      --prompt_style='gemma' \
      --evaluator='key_word' \
      --save_path='logs/fine-tuning-attack/safety_eval/gemma_1.1_7b_pure_bad_sft.json' \
      --eval_template='pure_bad' ;


accelerate launch  --num_processes=4 \
  eval_safety.py --model_name_or_path='ckpts/gemma-1.1-7b-it' \
      --torch_dtype=bfloat16 \
      --safety_bench='hex-phi' \
      --model_family='gemma' \
      --prompt_style='gemma' \
      --evaluator='key_word' \
      --save_path='logs/fine-tuning-attack/safety_eval/gemma_1.1_7b_pure_bad_sft.json' \
      --eval_template='pure_bad' ;