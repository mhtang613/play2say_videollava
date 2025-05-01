
JSON_FOLDER="/home/ec2-user/Video-LLaVA/playground/Video-LLaVA/train_json"
IMAGE_FOLDER="/home/ec2-user/Video-LLaVA/playground/Video-LLaVA"
VIDEO_FOLDER="/home/ec2-user/Video-LLaVA/playground/Video-LLaVA"

cd /home/ec2-user/Video-LLaVA
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 deepspeed videollava/train/train_mem.py \
    --deepspeed ./scripts/zero2.json \
     --model_name_or_path /home/ec2-user/Video-LLaVA/Tiny-Vicuna-1B \
    --version v1 \
    --data_path ${JSON_FOLDER}/llava_image_.json ${JSON_FOLDER}/valley_.json \
    --image_folder ${IMAGE_FOLDER} \
    --image_tower /home/ec2-user/Video-LLaVA/LanguageBind_Image \
    --video_folder ${VIDEO_FOLDER} \
    --video_tower /home/ec2-user/Video-LLaVA/LanguageBind_Video_merge \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 True \
    --output_dir ./checkpoints/videollava-7b-pretrain \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 24000 \
    --save_total_limit 1 \
    --learning_rate 1e-3 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048  --tokenizer_model_max_length 3072 \
    --gradient_checkpointing True \
    --dataloader_num_workers 1 \
    --lazy_preprocess True \
    --report_to tensorboard \
    --cache_dir "./cache_dir"
