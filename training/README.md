The code is built on top of [Movement Pruning](https://github.com/huggingface/transformers/tree/master/examples/research_projects/movement-pruning).

## Usage

### Requirements
Please install all the dependencies:
```
pip install -r requirements.txt
```

Then, please download the necessary dataset you need by yourself, e.g., QQP and MNLI. 

### Training
Afterward, you can train your own pruned model! First, for full model training:
```
export PYTHONUNBUFFERED=1

OUTPUT_PATH=result/qqp_partial/1.0

mkdir -p ${OUTPUT_PATH}

export CUDA_VISIBLE_DEVICES=3; python masked_run_glue.py --output_dir ${OUTPUT_PATH} --data_dir ../../data-bin/glue_data/QQP \
--task_name qqp --do_train --do_eval --do_lower_case --model_type bert --model_name_or_path bert-base-uncased \
--per_gpu_train_batch_size 32 --overwrite_output_dir --warmup_steps 11000 --num_train_epochs 10 \
--max_seq_length 128 --learning_rate 3e-05 --mask_scores_learning_rate 1e-2 \
--evaluate_during_training --logging_steps 500 --save_steps 500 --fp16 \
--final_threshold 1.0  --head_pruning --final_lambda 3000 --pruning_method topK --mask_init constant \
--mask_scale 0. | tee -a ${OUTPUT_PATH}/training_log.txt 
```

For head/row pruning training, we use the model trained from full model training as the teacher model (as path as above, you can replace it with your pre-trained model):
```
export PYTHONUNBUFFERED=1

threshold=0.5

teacher_path=result/qqp_partial/1.0/checkpoint-110000/ 

OUTPUT_PATH=result/qqp_partial/${threshold}

mkdir -p ${OUTPUT_PATH}

export CUDA_VISIBLE_DEVICES=3; python masked_run_glue.py --output_dir ${OUTPUT_PATH} --data_dir ../../data-bin/glue_data/QQP \
--task_name qqp --do_train --do_eval --do_lower_case --model_type masked_bert \
--model_name_or_path bert-base-uncased --per_gpu_train_batch_size 32 --overwrite_output_dir \
--warmup_steps 11000 --num_train_epochs 20 --max_seq_length 128 \
--learning_rate 3e-05 --mask_scores_learning_rate 1e-2 --evaluate_during_training \
--logging_steps 11000 --save_steps 11000 \
--teacher_type bert --teacher_name_or_path  ${teacher_path} \
--fp16 --final_threshold ${threshold} --head_pruning --final_lambda 20000 --pruning_method topK \
--mask_init constant --mask_scale 0. | tee -a ${OUTPUT_PATH}/training_log.txt 
```

For block-wise pruning training, we use the previously trained model (aka head/row pruned model)  as the teacher (as path as above, you can replace it with your pre-trained model):
```
export PYTHONUNBUFFERED=1

threshold=0.3

teacher_path=result/qqp_partial/0.5/checkpoint-209000/ 

block_rows=16
block_cols=16

OUTPUT_PATH=result/qqp_full/${block_rows}_${block_cols}/${threshold}

mkdir -p ${OUTPUT_PATH}

export CUDA_VISIBLE_DEVICES=3; python masked_blockwise_run_glue.py --output_dir ${OUTPUT_PATH} --data_dir ../../data-bin/glue_data/QQP \
--task_name qqp --do_train --do_eval --do_lower_case --model_type masked_bert \
--model_name_or_path ${teacher_path} --per_gpu_train_batch_size 32 --overwrite_output_dir \
--warmup_steps 11000 --num_train_epochs 20 --max_seq_length 128 --block_rows ${block_rows} --block_cols ${block_cols} \
--learning_rate 3e-05 --mask_scores_learning_rate 1e-2  --evaluate_during_training \
--logging_steps 11000 --save_steps 11000 --teacher_type masked_bert --teacher_name_or_path ${teacher_path} \
--fp16 --final_threshold ${threshold} --final_lambda 20000 --pruning_method topK \
--mask_init constant --mask_scale 0. | tee -a ${OUTPUT_PATH}/train_log.txt
```

### Parameter Count 
We also provide a script to count the remaining parameters after head/row pruning as well as full MLPruning. Note that this script can also be used to measure the speedup of the model after head/row pruning since all MatMuls are dense ones. 

For the head/row pruning model, please use the following command to compute the remaining parameters and the speedup:
```
OUTPUT_PATH=result/qqp_partial/0.5/checkpoint-209000/
bs=1
seq=128
export CUDA_VISIBLE_DEVICES=3; python masked_bert_parameter_count.py \
                            --model_type masked_bert \
                            --model_name_or_path ${OUTPUT_PATH}  \
                            --per_gpu_train_batch_size ${bs} \
                            --max_seq_length ${seq} \
                            --pruning_method topK \
                            --block_cols -1 --block_rows -1 \
                            --head_pruning
```

For models after MLPruning, please use the following command to compute the remaining parameters (the output still has throughput but this does not have block sparse kernel supporting. In order to test the speedup, please see xxx)
```
T_OUTPUT_PATH=result/qqp_partial/0.5/checkpoint-209000/ # used to get row/head pruning
bs=1
seq=128
OUTPUT_PATH=result/qqp_full/16_16/0.3/checkpoint-11000/
export CUDA_VISIBLE_DEVICES=3; python masked_bert_parameter_count.py \
                            --model_type masked_bert \
                            --model_name_or_path ${T_OUTPUT_PATH}  \
                            --block_path ${OUTPUT_PATH} \
                            --per_gpu_train_batch_size ${bs} \
                            --max_seq_length ${seq} \
                            --pruning_method topK \
                            --block_cols 16 --block_rows 16 \
                            --head_pruning
```

## TODO
Add SQuAD related running script.