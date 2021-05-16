# This code is modified from https://github.com/huggingface/transformers/tree/master/examples/pytorch/text-classification
# Licensed under the Apache License, Version 2.0 (the "License");
# We add more functionalities as well as remove unnecessary functionalities

import argparse
import glob
import json
import logging
import os
import random
import math 

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from emmental import MaskedBertConfig, MaskedBertForSequenceClassification, MaskedLinear
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

scaler = torch.cuda.amp.GradScaler()

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

def regularization(model: nn.Module, threshold: float):
    threshold_list = []
    score_list = []
    for name, param in model.named_parameters():
        if 'threshold' in name:
            threshold_list.append(torch.sigmoid(param))
        if 'score' in name:
            score_list.append(param.numel())

    total_num = sum(score_list)
    param_remain = 0
    
    for i, j in zip(threshold_list, score_list):
        param_remain += i * j

    if param_remain / total_num - threshold <= 0:
        reg_loss = param_remain * 0.
    else:
        reg_loss = torch.square(param_remain / total_num - threshold) # 144 comes from count, use simple sqaure loss

    return reg_loss


def train(args, train_dataset, model, tokenizer, teacher=None):
    """ Train the model """
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // len(train_dataloader) + 1
    else:
        t_total = len(train_dataloader) * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ["bias", "LayerNorm.weight"]

    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if "mask_score" in n or "threshold" in n and p.requires_grad],
            "lr": args.mask_scores_learning_rate,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n and "threshold" not in n and p.requires_grad and not any(nd in n for nd in no_decay)
            ],
            "lr": args.learning_rate,
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if "mask_score" not in n and "threshold" not in n and p.requires_grad and any(nd in n for nd in no_decay)
            ],
            "lr": args.learning_rate,
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True,
        )

    # Train!
    print("***** Running training *****")
    print(f"  Num examples = {len(train_dataset)}")
    print(f"  Num Epochs = {args.num_train_epochs}")
    print(f"  Instantaneous batch size per GPU = {args.per_gpu_train_batch_size}")
    print(
        f"  Total train batch size (w. parallel, distributed) = {args.train_batch_size}",
        
    )
    print(f"  Total optimization steps = {t_total}")
    # Distillation
    if teacher is not None:
        print("  Training with distillation")

    global_step = 0
    epochs_trained = 0
    steps_trained_in_current_epoch = 0
    # Check if continuing training from a checkpoint

    tr_loss, logging_loss = 0.0, 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained,
        int(args.num_train_epochs),
        desc="Epoch",
        disable=args.local_rank not in [-1, 0],
    )
    set_seed(args)  # Added here for reproducibility

    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):

            # Skip past any already trained steps if resuming training
            if steps_trained_in_current_epoch > 0:
                steps_trained_in_current_epoch -= 1
                continue

            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
            if args.model_type != "distilbert":
                inputs["token_type_ids"] = (
                    batch[2] if args.model_type in ["bert", "masked_bert", "xlnet", "albert"] else None
                )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids

            
            with torch.cuda.amp.autocast(enabled=args.fp16):
                outputs = model(**inputs)
                # print(outputs)
                
                if "masked" not in args.model_type:
                    loss, logits_stu = outputs.loss, outputs.logits  # model outputs are always tuple in transformers (see doc)
                else:
                    loss, logits_stu, reps_stu, attentions_stu = outputs

                # Distillation loss
                if teacher is not None:
                    if "token_type_ids" not in inputs:
                        inputs["token_type_ids"] = None if args.teacher_type == "xlm" else batch[2]
                    with torch.no_grad():
                        outputs_tea = teacher(
                            input_ids=inputs["input_ids"],
                            token_type_ids=inputs["token_type_ids"],
                            attention_mask=inputs["attention_mask"],
                        )
                        if "masked" not in args.teacher_type:
                            logits_tea, reps_tea, attentions_tea = outputs_tea.logits, outputs_tea.hidden_states, outputs_tea.attentions
                        else:
                            logits_tea, reps_tea, attentions_tea = outputs_tea
                         
                        
                    teacher_layer_num = len(attentions_tea)
                    student_layer_num = len(attentions_stu)
                    assert teacher_layer_num % student_layer_num == 0
                    layers_per_block = int(teacher_layer_num / student_layer_num)
                    new_attentions_tea = [attentions_tea[i * layers_per_block + layers_per_block - 1]
                                        for i in range(student_layer_num)]

                    att_loss, rep_loss = 0, 0
                    if "masked" in args.teacher_type:
                        for student_att, teacher_att in zip(attentions_stu, new_attentions_tea):
                            student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(args.device),
                                                    student_att)
                            teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(args.device),
                                                    teacher_att)

                            tmp_loss = F.mse_loss(student_att, teacher_att, reduction="mean",)
                            att_loss += tmp_loss

                    new_reps_tea = [reps_tea[i * layers_per_block] for i in range(student_layer_num + 1)]
                    new_reps_stu = reps_stu
                    for i_threp, (student_rep, teacher_rep) in enumerate(zip(new_reps_stu, new_reps_tea)):
                        tmp_loss = F.mse_loss(student_rep, teacher_rep, reduction="mean",)
                        rep_loss += tmp_loss

                    loss_logits = F.kl_div(
                            input=F.log_softmax(logits_stu / args.temperature, dim=-1),
                            target=F.softmax(logits_tea / args.temperature, dim=-1),
                            reduction="batchmean",
                        ) * (args.temperature ** 2)
                    
                    loss_distill = loss_logits + rep_loss + att_loss

                    loss = args.alpha_distil * loss_distill + args.alpha_ce * loss

            regu_ = regularization(model=model, threshold=args.final_threshold)
            regu_lambda = max(args.final_lambda * regu_.item() / (1 - args.final_threshold) / (1 - args.final_threshold), 50)
            if regu_.item() < 0.0003:
                # when the loss is very small, no need to pubnish it too much
                regu_lambda = 10. 

            loss = loss + regu_lambda * regu_


            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training


            if args.fp16:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tr_loss += loss.item()
            if True:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                if args.fp16:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logs = {}
                    if (
                        args.local_rank == -1 and args.evaluate_during_training
                    ):  # Only evaluate when single GPU otherwise metrics may not average well
                        results = evaluate(args, model, tokenizer)
                        for key, value in results.items():
                            eval_key = "eval_{}".format(key)
                            logs[eval_key] = value
                            
                    loss_scalar = (tr_loss - logging_loss) / args.logging_steps
                    learning_rate_scalar = scheduler.get_lr()
                    logs["learning_rate"] = learning_rate_scalar[0]
                    if len(learning_rate_scalar) > 1:
                        for idx, lr in enumerate(learning_rate_scalar[1:]):
                            logs[f"learning_rate/{idx+1}"] = lr
                    logs["loss"] = loss_scalar
                    if teacher is not None:
                        logs["loss/distil"] = loss_distill.item()
                        logs["loss/distil_logits"] = loss_logits.item()
                        try:
                            logs["loss/distil_attns"] = att_loss.item()
                        except:
                            logs["loss/distil_attns"] = 0
                        try:
                            logs["loss/distil_reps"] = rep_loss.item()
                        except:
                            logs["loss/distil_reps"] = 0
                        
                    print(f"step: {global_step}: {logs}")

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    # Save model checkpoint
                    output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = (
                        model.module if hasattr(model, "module") else model
                    )  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)

                    torch.save(args, os.path.join(output_dir, "training_args.bin"))
                    print(f"Saving model checkpoint to {output_dir}")

                    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                    if args.fp16:
                        torch.save(scaler.state_dict(), os.path.join(output_dir, "scaler.pt"))
                        
                    print(f"Saving optimizer and scheduler states to {output_dir}")

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break
        if args.max_steps > 0 and global_step > args.max_steps:
            train_iterator.close()
            break



    print('Best Result: ', best_accuracy)

    return global_step, tr_loss / global_step


def evaluate(args, model, tokenizer, prefix=""):
    # Loop to handle MNLI double evaluation (matched, mis-matched)
    eval_task_names = ("mnli", "mnli-mm") if args.task_name == "mnli" else (args.task_name,)
    eval_outputs_dirs = (args.output_dir, args.output_dir + "/MM") if args.task_name == "mnli" else (args.output_dir,)

    results = {}
    for eval_task, eval_output_dir in zip(eval_task_names, eval_outputs_dirs):
        eval_dataset = load_and_cache_examples(args, eval_task, tokenizer, evaluate=True)

        if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
            os.makedirs(eval_output_dir)

        args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
        # Note that DistributedSampler samples randomly
        eval_sampler = SequentialSampler(eval_dataset)
        eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

        # multi-gpu eval
        if args.n_gpu > 1 and not isinstance(model, torch.nn.DataParallel):
            model = torch.nn.DataParallel(model)

        # Eval!
        # print(f"***** Running evaluation {prefix} *****")
        # print(f"  Num examples = {len(eval_dataset)}")
        # print(f"  Batch size = {args.eval_batch_size}")
        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            model.eval()
            batch = tuple(t.to(args.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if args.model_type in ["bert", "masked_bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                
                outputs = model(**inputs)
                
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        if args.output_mode == "classification":
            from scipy.special import softmax

            probs = softmax(preds, axis=-1)
            entropy = np.exp((-probs * np.log(probs)).sum(axis=-1).mean())
            preds = np.argmax(preds, axis=1)
        elif args.output_mode == "regression":
            entropy = None
            preds = np.squeeze(preds)
        result = compute_metrics(eval_task, preds, out_label_ids)
        results.update(result)
        if entropy is not None:
            result["eval_avg_entropy"] = entropy

        output_eval_file = os.path.join(eval_output_dir, prefix, "eval_results.txt")

    return results

def load_and_cache_examples(args, task, tokenizer, evaluate=False):
    if args.local_rank not in [-1, 0] and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    processor = processors[task]()
    output_mode = output_modes[task]
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(
        args.data_dir,
        "cached_{}_{}_{}_{}".format(
            "dev" if evaluate else "train",
            list(filter(None, args.model_name_or_path.split("/"))).pop(),
            str(args.max_seq_length),
            str(task),
        ),
    )
    if os.path.exists(cached_features_file) and not args.overwrite_cache:
        features = torch.load(cached_features_file)
    else:
        label_list = processor.get_labels()
        if task in ["mnli", "mnli-mm"] and args.model_type in ["roberta", "xlmroberta"]:
            # HACK(label indices are swapped in RoBERTa pretrained model)
            label_list[1], label_list[2] = label_list[2], label_list[1]
        examples = (
            processor.get_dev_examples(args.data_dir) if evaluate else processor.get_train_examples(args.data_dir)
        )
        features = convert_examples_to_features(
            examples,
            tokenizer,
            max_length=args.max_seq_length,
            label_list=label_list,
            output_mode=output_mode,
        )
        if args.local_rank in [-1, 0]:
            torch.save(features, cached_features_file)

    if args.local_rank == 0 and not evaluate:
        torch.distributed.barrier()  # Make sure only the first process in distributed training process the dataset, and the others will use the cache

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    if output_mode == "classification":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_labels = torch.tensor([f.label for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_labels)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument(
        "--data_dir",
        default=None,
        type=str,
        required=True,
        help="The input data dir. Should contain the .tsv files (or other data files) for the task.",
    )
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
    parser.add_argument(
        "--task_name",
        default=None,
        type=str,
        required=True,
        help="The name of the task to train selected in the list: " + ", ".join(processors.keys()),
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    # Other parameters
    parser.add_argument(
        "--config_name",
        default="",
        type=str,
        help="Pretrained config name or path if not the same as model_name",
    )
    parser.add_argument(
        "--tokenizer_name",
        default="",
        type=str,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--cache_dir",
        default="",
        type=str,
        help="Where do you want to store the pre-trained models downloaded from huggingface.co",
    )
    parser.add_argument(
        "--max_seq_length",
        default=128,
        type=int,
        help="The maximum total input sequence length after tokenization. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument("--do_eval", action="store_true", help="Whether to run eval on the dev set.")
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )
    parser.add_argument(
        "--do_lower_case",
        action="store_true",
        help="Set this flag if you are using an uncased model.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=32,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")

    # Pruning parameters
    parser.add_argument(
        "--mask_scores_learning_rate",
        default=1e-2,
        type=float,
        help="The Adam initial learning rate of the mask scores.",
    )
    parser.add_argument(
        "--final_threshold", default=0.7, type=float, help="Final value of the threshold (for scheduling)."
    )


    parser.add_argument(
        "--pruning_method",
        default="topK",
        type=str,
        help="Pruning Method (topK = MLPruning).",
    )
    parser.add_argument(
        "--mask_init",
        default="constant",
        type=str,
        help="Initialization method for the mask scores. Choices: constant, uniform, kaiming.",
    )
    parser.add_argument(
        "--mask_scale", default=0.0, type=float, help="Initialization parameter for the chosen initialization method."
    )

    parser.add_argument(
        "--final_lambda",
        default=0.0,
        type=float,
        help="Regularization intensity (used in conjunction with `regularization`.",
    )

    # Distillation parameters (optional)
    parser.add_argument(
        "--teacher_type",
        default=None,
        type=str,
        help="Teacher type. Teacher tokenizer and student (model) tokenizer must output the same tokenization. Only for distillation.",
    )
    parser.add_argument(
        "--teacher_name_or_path",
        default=None,
        type=str,
        help="Path to the already fine-tuned teacher model. Only for distillation.",
    )
    parser.add_argument(
        "--alpha_ce", default=0.1, type=float, help="Cross entropy loss linear weight. Only for distillation."
    )
    parser.add_argument(
        "--alpha_distil", default=0.9, type=float, help="Distillation loss linear weight. Only for distillation."
    )
    parser.add_argument(
        "--temperature", default=2.0, type=float, help="Distillation temperature. Only for distillation."
    )

    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--num_train_epochs",
        default=3.0,
        type=float,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--max_steps",
        default=-1,
        type=int,
        help="If > 0: set total number of training steps to perform. Override num_train_epochs.",
    )
    parser.add_argument("--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps.")

    parser.add_argument("--logging_steps", type=int, default=50, help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=1000, help="Save checkpoint every X updates steps.")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument(
        "--overwrite_cache",
        action="store_true",
        help="Overwrite the cached training and evaluation sets",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--block_rows", type=int, default=32, help="Number of rows in a block")
    parser.add_argument("--block_cols", type=int, default=32, help="Number of cols in a block")

    args = parser.parse_args()

    if (
        os.path.exists(args.output_dir)
        and os.listdir(args.output_dir)
        and args.do_train
        and not args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

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

    # Prepare GLUE task
    args.task_name = args.task_name.lower()
    if args.task_name not in processors:
        raise ValueError("Task not found: %s" % (args.task_name))
    processor = processors[args.task_name]()
    args.output_mode = output_modes[args.task_name]
    label_list = processor.get_labels()
    num_labels = len(label_list)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    args.model_type = args.model_type.lower()
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_labels,
        finetuning_task=args.task_name,
        cache_dir=args.cache_dir if args.cache_dir else None,
        pruning_method=args.pruning_method,
        mask_init=args.mask_init,
        mask_scale=args.mask_scale,
        output_attentions=True,
        output_hidden_states=True,
    )
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        cache_dir=args.cache_dir if args.cache_dir else None,
        do_lower_case=args.do_lower_case,
    )
    model = model_class.from_pretrained(
        args.model_name_or_path,
        from_tf=bool(".ckpt" in args.model_name_or_path),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )

    def make_block_pruning(model):

        # we need to do a evaluation to see the performance matches!!!!
        if 'mask' in args.model_type:
            model._make_structural_pruning([args.block_rows, args.block_cols])

        # add block-wise pruning part
        for module in model.modules():
            if isinstance(module, MaskedLinear):
                module.enable_block_pruning([args.block_rows, args.block_cols])
        return model 
    model = make_block_pruning(model)


    if args.teacher_type is not None:
        assert args.teacher_name_or_path is not None
        assert args.alpha_distil > 0.0
        assert args.alpha_distil + args.alpha_ce > 0.0
        teacher_config_class, teacher_model_class, _ = MODEL_CLASSES[args.teacher_type]
        teacher_config = teacher_config_class.from_pretrained(args.teacher_name_or_path)
        teacher_config.output_attentions = True
        teacher_config.output_hidden_states = True
        teacher = teacher_model_class.from_pretrained(
            args.teacher_name_or_path,
            from_tf=False,
            config=teacher_config,
            cache_dir=args.cache_dir if args.cache_dir else None,
        )
        if 'mask' in args.teacher_type:
            teacher._make_structural_pruning([None, None])
        teacher.to(args.device)
        # result = evaluate(args, teacher, tokenizer, prefix="")
        # print('Teacher Acc: ', result)
    else:
        teacher = None

    if args.local_rank == 0:
        torch.distributed.barrier()  # Make sure only the first process in distributed training will download model & vocab

    model.to(args.device)
    print(model)
    print(f"Training/evaluation parameters {args}")

    # Training
    if args.do_train:
        train_dataset = load_and_cache_examples(args, args.task_name, tokenizer, evaluate=False)
        global_step, tr_loss = train(args, train_dataset, model, tokenizer, teacher=teacher)

    # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
    if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = (
            model.module if hasattr(model, "module") else model
        )  # Take care of distributed/parallel training
        model_to_save.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(args, os.path.join(args.output_dir, "training_args.bin"))

        tmp_result = evaluate(args, model, tokenizer, prefix="")
        print(f"Final: {tmp_result}")

if __name__ == "__main__":
    main()
