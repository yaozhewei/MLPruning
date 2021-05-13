# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Fine-pruning Masked BERT on sequence classification on GLUE."""

import argparse
import glob
import json
import logging
import os
import sys
import random
import time 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from emmental import MaskedBertConfig, MaskedBertForSequenceClassification, MaskedLinear, MaskedBertForQuestionAnswering
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)
from transformers import glue_compute_metrics as compute_metrics
from transformers import glue_convert_examples_to_features as convert_examples_to_features
from transformers import glue_output_modes as output_modes
from transformers import glue_processors as processors



logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "bert": (BertConfig, BertForSequenceClassification, BertTokenizer),
    "masked_bert": (MaskedBertConfig, MaskedBertForSequenceClassification, BertTokenizer),
}

def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--model_type",
        default=None,
        type=str,
        required=True,
        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()),
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models",
    )
    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--pruning_method",
        default="topK",
        type=str,
        help="Pruning Method (l0 = L0 regularization, magnitude = Magnitude pruning, topK = Movement pruning, sigmoied_threshold = Soft movement pruning).",
    )
    parser.add_argument(
        "--head_pruning", action="store_true", help="Head Pruning or not",
    )
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")    
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    parser.add_argument("--block_rows", type=int, default=-1, help="Number of rows in a block")
    parser.add_argument("--block_cols", type=int, default=-1, help="Number of cols in a block")
    parser.add_argument(
        "--block_path",
        default=None,
        type=str,
        help="Path to pretrained block wise model",
    )
    
    args = parser.parse_args()

    

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of synchronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl")
        args.n_gpu = 1
    args.device = device

    # Set seed
    set_seed(args)

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    if 'qqp' in args.model_name_or_path or 'mnli' in args.model_name_or_path:
        num_labels = 2  
        if 'mnli' in args.model_name_or_path:
            num_labels = 3
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_labels,
            finetuning_task='mrpc',
            cache_dir= None,
            pruning_method=args.pruning_method,
            mask_init='constant',
            mask_scale=0,
            head_pruning=args.head_pruning
        )
    elif 'squad' in args.model_name_or_path:
        print('This one is used!')
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            cache_dir= None,
            pruning_method=args.pruning_method,
            mask_init='constant',
            mask_scale=0,
            head_pruning = args.head_pruning
        )
        model_class = MaskedBertForQuestionAnswering
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir= None,
        )
    model.eval()

    if args.block_path is None:
        model._make_structural_pruning([None, None])
    else:
        assert args.block_rows >= 1 and args.block_cols >= 1
        model._make_structural_pruning([args.block_rows, args.block_cols])
        for module in model.modules():
            if isinstance(module, MaskedLinear):
                module.enable_block_pruning([args.block_rows, args.block_cols])
        model.load_state_dict(torch.load(f"{args.block_path}/pytorch_model.bin"))
        for module in model.modules():
            if isinstance(module, MaskedLinear):
                module.make_block_wise_inference_pruning() # block-sparse model 

                
    total_num_params = 0
    for name, param in model.named_parameters():
        if 'encoder' in name:
            total_num_params += (param.abs() > 1e-8).sum()
    

    model.to(args.device)
    
    batch_size = args.per_gpu_train_batch_size
    length = args.max_seq_length
    batch = {
        "attention_mask": torch.ones([batch_size, length], dtype=torch.long).cuda(),
        "input_ids": torch.ones([batch_size, length], dtype=torch.long).cuda(),
        "token_type_ids": torch.ones([batch_size, length], dtype=torch.long).cuda(),
    }
    inputs = {"input_ids": batch["input_ids"], "attention_mask": batch["attention_mask"]}
    
    if args.model_type != "distilbert":
        inputs["token_type_ids"] = (
            batch["token_type_ids"] if args.model_type in ["bert", "masked_bert", "xlnet", "albert"] else None
        )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

    # warmup!!!
    for i in range(10):    
        with torch.no_grad():
            outputs = model(**inputs)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    # do real measurement 
    num_runs = 100
    torch.cuda.synchronize()
    start.record()
    for i in range(num_runs):
        with torch.no_grad():    
            outputs = model(**inputs)
    end.record()
    torch.cuda.synchronize()

    total_time = start.elapsed_time(end) / 1000 # s
    print('*' * 100)
    print('Num of Parameters: ', total_num_params.item())
    print(f'Remaining Parameters as compared to baseline: {(total_num_params/85054608*100):.2f}%')
    print(f"{num_runs/total_time * batch_size} Sentences / s")
    print(f"{total_time/num_runs/batch_size * 1000} ms / Sentences ")
    print('*' * 100)



if __name__ == "__main__":
    main()
