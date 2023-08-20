log=./log_files/log_final_llama2_cutoff_exp.txt

data_dir=./sample_data/2018n2c2_aiu_th3/cutoff_0/
nmd=/vol/bitbucket/l22/final_llama2_models_cutoff0
pof=./predictions_final_llama2_cutoff0.txt
ckpt_path=/vol/bitbucket/l22/final_llama2_models_cutoff0/ckpt_0

accelerate launch ./src/relation_extraction.py \
                --model_type llama \
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
                --num_train_epochs 3 \
                --gradient_accumulation_steps 4 \
                --do_warmup \
                --warmup_ratio 0.1 \
                --weight_decay 0 \
                --max_num_checkpoints 2 \
                --log_file $log \
                --log_step 500 \
                --progress_bar

accelerate launch ./src/relation_extraction.py \
		--model_type llama \
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

data_dir=./sample_data/2018n2c2_aiu_th3/cutoff_1/
nmd=/vol/bitbucket/l22/final_llama2_models_cutoff1
pof=./predictions_final_llama2_cutoff1.txt
ckpt_path=/vol/bitbucket/l22/final_llama2_models_cutoff1/ckpt_0

accelerate launch ./src/relation_extraction.py \
                --model_type llama \
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
                --num_train_epochs 3 \
                --gradient_accumulation_steps 4 \
                --do_warmup \
                --warmup_ratio 0.1 \
                --weight_decay 0 \
                --max_num_checkpoints 2 \
                --log_file $log \
                --log_step 500 \
                --progress_bar

accelerate launch ./src/relation_extraction.py \
		--model_type llama \
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

data_dir=./sample_data/2018n2c2_aiu_th3/cutoff_2/
nmd=/vol/bitbucket/l22/final_llama2_models_cutoff2
pof=./predictions_final_llama2_cutoff2.txt
ckpt_path=/vol/bitbucket/l22/final_llama2_models_cutoff2/ckpt_0

accelerate launch ./src/relation_extraction.py \
                --model_type llama \
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
                --num_train_epochs 3 \
                --gradient_accumulation_steps 4 \
                --do_warmup \
                --warmup_ratio 0.1 \
                --weight_decay 0 \
                --max_num_checkpoints 2 \
                --log_file $log \
                --log_step 500 \
                --progress_bar

accelerate launch ./src/relation_extraction.py \
		--model_type llama \
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

data_dir=./sample_data/2018n2c2_aiu_th3/cutoff_3/
nmd=/vol/bitbucket/l22/final_llama2_models_cutoff3
pof=./predictions_final_llama2_cutoff3.txt
ckpt_path=/vol/bitbucket/l22/final_llama2_models_cutoff3/ckpt_0

accelerate launch ./src/relation_extraction.py \
                --model_type llama \
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
                --num_train_epochs 3 \
                --gradient_accumulation_steps 4 \
                --do_warmup \
                --warmup_ratio 0.1 \
                --weight_decay 0 \
                --max_num_checkpoints 2 \
                --log_file $log \
                --log_step 500 \
                --progress_bar

accelerate launch ./src/relation_extraction.py \
		--model_type llama \
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