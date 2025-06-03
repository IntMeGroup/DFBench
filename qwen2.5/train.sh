CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift sft \
    --model 'Qwen/Qwen2.5-VL-7B-Instruct' \
    --train_type lora \
    --dataset './datasets/img_train.json' \
    --val_dataset './datasets/img_test.json' \
    --split_dataset_ratio 0 \
    --torch_dtype bfloat16 \
    --freeze_vit True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 1 \
    --eval_steps 64645 \
    --save_steps 12929 \
    --save_total_limit 5 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output_ckpt \
    --system 'You are a helpful assistant.' \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4 \
    --model_author swift \
    --model_name swift-robot

# --model_kwargs '{"FPS": 2}'
# --freeze_llm 
