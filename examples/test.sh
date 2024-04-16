#!/bin/bash
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export LPAI_INPUT_DATA_0=/lpai/volumes/lpai-demo-muses
export CKPT_DIR=$LPAI_INPUT_DATA_0/megatron_pretrain/ckpt-dongwei
export LATEST_CKPT_FILE=$CKPT_DIR/latest_checkpointed_iteration.txt
export LPAI_INPUT_MODEL_0=/lpai/megatron-lm/models/model_sp_28-06-2023-06-57.model_210102_new_239321_new_256000_new
export DATA_CACHE_DIR=$LPAI_INPUT_DATA_0/megatron_pretrain/data-cache-dongwei
mkdir -p $DATA_CACHE_DIR

if [[ -f $LATEST_CKPT_FILE ]]; then
    content=$(cat "$LATEST_CKPT_FILE" | tr -d '[:space:]')
    if [[ $content =~ ^[0-9]+$ ]]; then
        export INIT_OR_NOT="--no-initialization"
        echo "resume from $LATEST_CKPT_FILE"
    else
        export INIT_OR_NOT=""
        echo "Latest ckpt does not exist."
    fi
else
    export INIT_OR_NOT=""
    echo "Ckpt does not exist."
fi
export LPAI_INPUT_DATASET_0=/lpai/dataset/pretrain-3t-tokens/0-1-0
export DS_BASE="${LPAI_INPUT_DATASET_0}"
export HIDDEN_SIZE=512
export ATTEN_HEADS=32
export NUM_LAYERS=10
export TRAIN_SAMPLES=800000000
export SEQ_LEN=4096
export TP=1
export PP=2
export MICRO_BS=1
export GLOBAL_BS=16
export LR=3e-4
export MIN_LR=3e-5
export DATASET_CC1="${DS_BASE}/merged_2020-50"
export DATASET_CC2="${DS_BASE}/merged_2021-21"
export DATASET_CC3="${DS_BASE}/merged_2021-31"
export DATASET_CC4="${DS_BASE}/merged_2022-21"
export DATASET_CC5="${DS_BASE}/merged_2022-33"
export DATASET_CC6="${DS_BASE}/merged_2022-49"
export DATASET_CC7="${DS_BASE}/merged_2023-14"
export DATASET_ARXIV="${DS_BASE}/merged_arxiv"
export DATASET_PILE_MED="${DS_BASE}/merged_pile-med"
export DATASET_REDDIT="${DS_BASE}/merged_reddit"
export DATASET_DOUBAN="${DS_BASE}/merged_douban"
export DATASET_STACK_EXCHANGE="${DS_BASE}/merged_stack_exchange"
export DATASET_WIKIPEDIA="${DS_BASE}/merged_wikipedia"
export DATASET_GITHUB="${DS_BASE}/merged_github-cleaned"
export DATASET_BOOKS3="${DS_BASE}/merged_books3-minus-gutenberg"
export DATASET_ZHIHU="${DS_BASE}/merged_zhihu"
export DATASET_GUTENBERG="${DS_BASE}/merged_gutenberg"
export DATASET_PILE_OF_LAW="${DS_BASE}/merged_pile-of-law"
export DATASET_FALCON="${DS_BASE}/merged_falcon_refinedweb"
export DATASET_GITHUB_ISSUES="${DS_BASE}/merged_github-issues"
export DATASET_WANJUAN_EXAM="${DS_BASE}/merged_wanjuan_exam_text"
export DATASET_WANJUAN_PATENT="${DS_BASE}/merged_wanjuan_patent_news"
export DATASET="0.087 ${DATASET_CC1} 0.087 ${DATASET_CC2} 0.087 ${DATASET_CC3} 0.087 ${DATASET_CC4} 0.087 ${DATASET_CC5} 0.087 ${DATASET_CC6} 0.087 ${DATASET_CC7} 0.025 ${DATASET_ARXIV} 0.011 ${DATASET_PILE_MED} 0.001 ${DATASET_REDDIT} 0.0004 ${DATASET_DOUBAN} 0.01 ${DATASET_STACK_EXCHANGE} 0.045 ${DATASET_WIKIPEDIA} 0.045 ${DATASET_GITHUB} 0.035 ${DATASET_BOOKS3} 0.0001 ${DATASET_ZHIHU} 0.01 ${DATASET_GUTENBERG} 0.02 ${DATASET_PILE_OF_LAW} 0.1715 ${DATASET_FALCON} 0.01 ${DATASET_GITHUB_ISSUES} 0.003 ${DATASET_WANJUAN_EXAM} 0.004 ${DATASET_WANJUAN_PATENT}"
export DISTRIBUTED_ARGS="
    --nnodes=1 \
    --nproc_per_node=2 \
    --max_restarts=0 \
    --rdzv_id=dongwei_test \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:6000 \
    --rdzv_conf timeout=1800
"

export DATA_ARGS="
    --num-workers 4 \
    --save $CKPT_DIR \
    --load $CKPT_DIR \
    --data-cache-path $DATA_CACHE_DIR \
    --data-path $DATASET \
    --split 995,4,1
"

export OUTPUT_ARGS="
    --save-interval 10 \
    --log-interval 10 \
    --eval-iters 20 \
    --eval-interval 400 \
    --tensorboard-dir /lpai/output/tensorboard \
    --log-validation-ppl-to-tensorboard \
    --log-world-size-to-tensorboard \
    --log-timers-to-tensorboard \
    --log-validation-ppl-to-tensorboard
"

export LLAMA2_ARGS="
    --bf16 \
    --transformer-impl transformer_engine \
    --use-flash-attn \
    --untie-embeddings-and-output-weights \
    --swiglu \
    --num-layers ${NUM_LAYERS} \
    --hidden-size ${HIDDEN_SIZE} \
    --ffn-hidden-size 13824 \
    --num-attention-heads ${ATTEN_HEADS} \
    --init-method-std 0.02 \
    --global-batch-size ${GLOBAL_BS} \
    --train-samples ${TRAIN_SAMPLES} \
    --attention-dropout 0 \
    --hidden-dropout 0 \
    --tokenizer-type GPTSentencePieceTokenizer \
    --tokenizer-model $LPAI_INPUT_MODEL_0 \
    --vocab-size 256000 \
    --seq-length $SEQ_LEN \
    --max-position-embeddings $SEQ_LEN \
    --untie-embeddings-and-output-weights \
    --position-embedding-type rope \
    --normalization RMSNorm \
    --disable-bias-linear \
    --no-bias-dropout-fusion \
    --attention-softmax-in-fp32
"

export OPTM_ARGS="
    --override-opt_param-scheduler \
    --use-distributed-optimizer \
    --lr-warmup-samples 2048000 \
    --lr ${LR} \
    --min-lr ${MIN_LR} \
    --lr-decay-style cosine \
    --clip-grad 1.0 \
    --norm-epsilon 1e-6 \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --adam-eps 1e-5
"

export PARALLEL_ARGS="
    --overlap-grad-reduce \
    --sequence-parallel \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --micro-batch-size ${MICRO_BS}
"

export DIST_ARGS="
    --distributed-timeout-minutes 30 \
    --distributed-backend nccl \
    --accumulate-allreduce-grads-in-fp32
"

PYTHONPATH=$PYTHONPATH:/lpai/dlrover:/lpai/dlrover/dlrover/proto torchrun $DISTRIBUTED_ARGS \
pretrain_gpt.py --seed 78365 \
        $DATA_ARGS \
        $OUTPUT_ARGS \
        $LLAMA2_ARGS \
        $OPTM_ARGS \
        $PARALLEL_ARGS \
        $DIST_ARGS \
        $PROFILE_ARGS \
        $INIT_OR_NOT
