# export CUDA_VISIBLE_DEVICES=1
data_dir=./sample_data
nmd=/vol/bitbucket/l22/llama2_trained_models
pof=./predictions_llama2_trained_models.txt
log=./log_llama2_trained_models.txt
ckpt_path=/vol/bitbucket/l22/llama2_trained_models/ckpt_0

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
		--train_batch_size 2 \
		--eval_batch_size 2 \
		--learning_rate 1e-4 \
		--num_train_epochs 5 \
		--gradient_accumulation_steps 4 \
		--do_warmup \
		--warmup_ratio 0.1 \
		--weight_decay 0 \
		--max_num_checkpoints 2 \
		--log_file $log \
		--log_step 500 \
		--progress_bar

accelerate launch ./src/relation_extraction.py \
		--model_type llama2 \
		--ckpt_dir $ckpt_path\
		--data_format_mode 0 \
		--classification_scheme 2 \
		--pretrained_model meta-llama/Llama-2-7b-hf \
		--data_dir $data_dir \
		--new_model_dir $nmd \
		--predict_output_file $pof \
		--seed 13 \
		--max_seq_length 256 \
		--cache_data \
		--do_predict \
		--do_lower_case \
		--eval_batch_size 2 \
		--log_file $log \
        --progress_bar
