#!/bin/bash

deepspeed --master_port your_port\
    lego/train/train.py\
    --deepspeed ./scripts/zero2.json\
    --model_name_or_path lmsys/vicuna-7b-v1.5 \
    --data_path path_to_your_dataset\
    --image_folder path_to_your_image_folder\
    --video_folder  path_to_your_video_folder\
    --sound_folder  path_to_your_sound_folder\
    --tune_mm_mlp_adapter True\
    --vision_tower openai/clip-vit-large-patch14-336\
    --mm_vision_select_layer -2\
    --mm_use_im_start_end True\
    --image_aspect_ratio pad\
    --bf16 True\
    --fp16 False\
    --output_dir  path_to_your_output_dir\
    --num_train_epochs 1\
    --per_device_train_batch_size 64\
    --per_device_eval_batch_size 4\
    --gradient_accumulation_steps 1\
    --evaluation_strategy "no"\
    --save_strategy "steps"\
    --save_steps 500\
    --save_total_limit 2\
    --learning_rate 1e-3\
    --weight_decay 0.\
    --warmup_ratio 0.03\
    --lr_scheduler_type cosine\
    --logging_steps 1\
    --tf32 True\
    --model_max_length 2048\
    --gradient_checkpointing True\
    --lazy_preprocess True\
    --report_to wandb