data_dir=./sample_data
nmd=/vol/bitbucket/l22/qkv_llama2_models_lora_16
pof=/homes/l22/Documents/git_repos/clini_dummy/qkv_predictions_llama2_lora_16.txt
log=/homes/l22/Documents/git_repos/clini_dummy/log_files/qkv_log_llama2_lora_exp.txt
ckpt_path=/vol/bitbucket/l22/qkv_llama2_models_lora_16/ckpt_0

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


data_dir=./sample_data
nmd=/vol/bitbucket/l22/qkv_llama2_models_lora_8
pof=/homes/l22/Documents/git_repos/clini_dummy/qkv_predictions_llama2_lora_8.txt
ckpt_path=/vol/bitbucket/l22/qkv_llama2_models_lora_8/ckpt_0

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



data_dir=./sample_data
nmd=/vol/bitbucket/l22/qkv_llama2_models_lora_4   
pof=/homes/l22/Documents/git_repos/clini_dummy/qkv_predictions_llama2_lora_4.txt
ckpt_path=/vol/bitbucket/l22/qkv_llama2_models_lora_4/ckpt_0


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

