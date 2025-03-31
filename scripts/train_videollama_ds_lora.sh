#!/bin/bash

# Environment Variables
ARG_WORLD_SIZE=${1:-1}
ARG_NPROC_PER_NODE=${1:-2}
ARG_MASTER_ADDR="127.0.0.1"
ARG_MASTER_PORT=16666
ARG_RANK=${3:-0}

# Multiple conditions
if [ ! -n "$WORLD_SIZE" ] || [ ! -n "$NPROC_PER_NODE" ]; then
    WORLD_SIZE=$ARG_WORLD_SIZE
    NPROC_PER_NODE=$ARG_NPROC_PER_NODE
fi
if [ ! -n "$MASTER_ADDR" ] || [ ! -n "$MASTER_PORT" ] || [ ! -n "$RANK" ]; then
    MASTER_ADDR=$ARG_MASTER_ADDR
    MASTER_PORT=$ARG_MASTER_PORT
    RANK=$ARG_RANK
fi

echo "WORLD_SIZE: $WORLD_SIZE"
echo "NPROC_PER_NODE: $NPROC_PER_NODE"


# Training Arguments
GLOBAL_BATCH_SIZE=2
LOCAL_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=$[$GLOBAL_BATCH_SIZE/($WORLD_SIZE*$NPROC_PER_NODE*$LOCAL_BATCH_SIZE)]

torchrun --nnodes $WORLD_SIZE \
    --nproc_per_node $NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    --node_rank $RANK \
    model/train_videollama.py \
    --deepspeed ds_config.json \
    --mamba_path "state-spaces/mamba-370m-hf" \
    --train_data "jsonl_TrainTesTval_noVidDesc_latest/train.jsonl" \
    --valid_data "jsonl_TrainTesTval_noVidDesc_latest/val.jsonl" \
    --model_dir "./mamba_compressor_log" \
    --llm_name "DAMO-NLP-SG/VideoLLaMA2.1-7B-AV" \
    --batch_size_single 1 \
    --batch_size_conv 1 \
    --epochs_single 3 \
    --epochs_conv 1 \
    --lr_single 2.5e-5 \
    --lr_conv 1e-4 \
    --eval_steps 100 \
    --patience_steps 2 \
    --gradient_accumulation_steps 1 \
    --scheduler_type reduce_on_plateau \
    --lora_r 4 \
    --lora_alpha 8 