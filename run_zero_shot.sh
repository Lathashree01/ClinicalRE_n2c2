#export CUDA_VISIBLE_DEVICES=1
#data_dir=/rds/general/user/l22/home/git_repos/thesis/data/2018n2c2_aio_th1/
data_dir=./sample_data
nmd=/vol/bitbucket/l22/zero_llama_1
pof=./zero_shot_llama1_predictions.txt
log=./zero_shot_llama1.txt
ckpt_dir=/vol/bitbucket/l22/llama_model/llama1

# we have to set data_dir, new_model_dir, model_type, log_file, and eval_batch_size, data_format_mode
python ./src/relation_extraction.py \
		--model_type llama1 \
		--data_format_mode 0 \
		--classification_scheme 2 \
                --ckpt_dir $ckpt_dir \
                --num_core 3 \
		--pretrained_model /vol/bitbucket/l22/llama_model/llama1 \
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
                --log_step 100 \
                --progress_bar
