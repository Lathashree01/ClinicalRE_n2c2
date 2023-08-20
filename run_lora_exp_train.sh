data_dir=./sample_data
nmd=/vol/bitbucket/l22/QKV_llama2_models_lora_16
pof=./QKV_predictions_llama2_lora_16.txt
log=./log_files/QKV_log_llama2_lora_exp.txt
ckpt_path=/vol/bitbucket/l22/QKV_llama2_models_lora_16/ckpt_0

accelerate launch ./src/relation_extraction.py \
                --model_type llama2 \
                --data_format_mode 0 \
                --classification_scheme 2 \
                --pretrained_model meta-llama/Llama-2-7b-hf  \
                --data_dir $data_dir \
                --num_core 3 \
                --new_model_dir $nmd \
                --predict_output_file $pof \
                --overwrite_model_dir \
                --seed 13 \
                --max_seq_length 256 \
                --cache_data \
                --do_train \
                --do_lower_case \
        		--lora_rank 16 \
        		--lora_alpha 32 \
                --train_batch_size 2 \
                --eval_batch_size 2 \
                --learning_rate 1e-4 \
                --num_train_epochs 3 \
                --gradient_accumulation_steps 4 \
                --do_warmup \
                --warmup_ratio 0.1 \
                --weight_decay 0 \
                --max_num_checkpoints 2 \
                --log_file $log \
                --log_step 500 \
                --progress_bar

data_dir=./sample_data
nmd=/vol/bitbucket/l22/QKV_llama2_models_lora_8
pof=./QKV_predictions_llama2_lora_8.txt
ckpt_path=/vol/bitbucket/l22/QKV_llama2_models_lora_8/ckpt_0

accelerate launch ./src/relation_extraction.py \
                --model_type llama2 \
                --data_format_mode 0 \
                --classification_scheme 2 \
                --pretrained_model meta-llama/Llama-2-7b-hf  \
                --data_dir $data_dir \
                --num_core 3 \
                --new_model_dir $nmd \
                --predict_output_file $pof \
                --overwrite_model_dir \
                --seed 13 \
                --max_seq_length 256 \
                --cache_data \
                --do_train \
                --do_lower_case \
                --lora_rank 8 \
                --lora_alpha 32 \
                --train_batch_size 2 \
                --eval_batch_size 2 \
                --learning_rate 1e-4 \
                --num_train_epochs 3 \
                --gradient_accumulation_steps 4 \
                --do_warmup \
                --warmup_ratio 0.1 \
                --weight_decay 0 \
                --max_num_checkpoints 2 \
                --log_file $log \
                --log_step 500 \
                --progress_bar

data_dir=./sample_data
nmd=/vol/bitbucket/l22/QKV_llama2_models_lora_4   
pof=./QKV_predictions_llama2_lora_4.txt
ckpt_path=/vol/bitbucket/l22/QKV_llama2_models_lora_4/ckpt_0

# NOTE: we have more options available, you can check our wiki for more information
accelerate launch ./src/relation_extraction.py \
                --model_type llama2 \
                --data_format_mode 0 \
                --classification_scheme 2 \
                --pretrained_model meta-llama/Llama-2-7b-hf  \
                --data_dir $data_dir \
                --num_core 3 \
                --new_model_dir $nmd \
                --predict_output_file $pof \
                --overwrite_model_dir \
                --seed 13 \
                --max_seq_length 256 \
                --cache_data \
                --do_train \
                --do_lower_case \
                --lora_rank 4 \
                --lora_alpha 32 \
                --train_batch_size 2 \
                --eval_batch_size 2 \
                --learning_rate 1e-4 \
                --num_train_epochs 3 \
                --gradient_accumulation_steps 4 \
                --do_warmup \
                --warmup_ratio 0.1 \
                --weight_decay 0 \
                --max_num_checkpoints 2 \
                --log_file $log \
                --log_step 500 \
                --progress_bar
