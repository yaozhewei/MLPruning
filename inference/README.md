## Usage

### Requirements
Please install all the dependencies:
```
pip install -r requirements.txt
```
You then need to install the latest nightly release of Triton. You can install from pip:
```
pip install -U --pre triton
```

After that, you can run the inference similar to the following example,
```
export PYTHONUNBUFFERED=1

OUTPUT_PATH=/home/ubuntu/str_prune/result/qqp_structural_distillation/0.5/acc_and_f1best/
BLOCK_PATH=/home/ubuntu/str_prune/result/qqp_structural_blockwise_32_distillation/0.4
batch_size=32
max_seq_length=512
pruning_method=structural*topK
block_size=32

export CUDA_VISIBLE_DEVICES=0; python masked_bert_parameter_count.py --model_type masked_bert \
--model_name_or_path ${OUTPUT_PATH} --per_gpu_train_batch_size ${batch_size} \
--max_seq_length ${max_seq_length} --pruning_method ${pruning_method} \
--block_cols ${block_size} --block_rows ${block_size} \
--block_path ${BLOCK_PATH} --head_pruning | tee -a ${OUTPUT_PATH}/param_log.txt
```