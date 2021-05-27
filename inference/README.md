Our sparse kenel is from [Triton](https://github.com/ptillet/triton). Please note that our inference code is tested on AWS g4dn.xlarge instance.	

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

OUTPUT_PATH=result/qqp_partial/0.5/checkpoint-209000/
block_rows=32
block_cols=32
BLOCK_PATH=result/qqp_full/${block_rows}_${block_cols}/0.4/checkpoint-209000/
batch_size=32
max_seq_length=512

export CUDA_VISIBLE_DEVICES=0; python masked_bert_inference.py --model_type masked_bert \
--model_name_or_path ${OUTPUT_PATH} --per_gpu_train_batch_size ${batch_size} \
--max_seq_length ${max_seq_length} --pruning_method topK \
--block_cols ${block_cols} --block_rows ${block_rows} \
--block_path ${BLOCK_PATH} --head_pruning
```
