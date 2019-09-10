# BERT Training with MXNet

## Experiment Setup

Script: https://github.com/dmlc/gluon-nlp/blob/master/scripts/bert/run_pretraining_hvd.py
Baseline: we use this paper from Google as baseline: https://arxiv.org/abs/1904.00962
It uses a two-stage pre-training. In the first stage, it uses batch size 512 with 128 tokens and trains 900k steps; in the second stage, it uses batch size 512 with 512 tokens and trains 100k steps.

We also follow the two-stage pre-training method. To calculate the number of steps in each stage, we apply the following formula:

nsteps = baseline_batch_size * num_tokens * num_steps / (batch_size * num_tokens)

Note that: in our run_pretraining_hvd.py script, the option —batch_size actually refers to batch_size * num_tokens per GPU. Besides, the total number of tokens should be multiplied by the argument —accumulation. 
Therefore, the divisor batch_size * num_tokens = script_batch_size * num_gpus * accumulation 

We look up hyperparameters (learning rate, warmup ratio) from this paper: https://arxiv.org/abs/1904.00962 based on the batch_size

In addition, we also change the following in script:

1. Changed optimizer from ‘bertadam’ to ‘lamb’
2. Added hyperparameter 'bias_correction': False 
3. --max_predictions_per_seq = max_seq_length * masked_lm_prob = 20
4. Use BERT-LARGE model: —model ‘bert_24_1024_16’
5. mpirun arguments:
    1. set map-by to ppr:4:socket
    2. set MXNET_SAFE_ACCUMULATION=1

## Parameters

### Stage 1
|Instance	|Num GPUs	|Comm	Tokens/GPU	|Batch Size	|Steps	|accumulation	|max_predictions_per_seq	|lr	|warmup_ratio	|
|---	|---	|---	|---	|---	|---	|---	|---	|---	|---	|---	|
|32 x p3dn (EFA) |256	|HVD	| 8192 |2048	|25600	|2	|20	|0.00354	|0.1	|
|4 x p3dn (EFA)	|32	|BytePS	| 8192	|2048	|225000	|1	|19.2	|0.00125	|0.0125	|
|1 x p3dn	|8	|BytePS	|8192	|256	|900000	|1	|	|1.00E-04	|	|

## Run

### Horovod
```
mpirun --allow-run-as-root --tag-output -np 32 --hostfile /home/ubuntu/efs/hosts  -map-by ppr:4:socket -mca pml ob1 -mca btl ^openib  -mca btl_tcp_if_include ens5 -x NCCL_DEBUG=INFO -x NCCL_SOCKET_IFNAME=ens5 \
        -x NCCL_MIN_NRINGS=8 -x HOROVOD_HIERARCHICAL_ALLREDUCE=1 \
        -x MXNET_SAFE_ACCUMULATION=1 \
        python run_pretraining_hvd.py \
                --model='bert_24_1024_16' \
                --data='/home/ubuntu/mxnet-data/bert-pretraining/datasets/book-corpus/book-corpus-large-split/*.train,/home/ubuntu/mxnet-data/bert-pretraining/datasets/enwiki/enwiki-feb-doc-split/*.train' \
                --data_eval='/home/ubuntu/mxnet-data/bert-pretraining/datasets/book-corpus/book-corpus-large-split/*.dev,/home/ubuntu/mxnet-data/bert-pretraining/datasets/enwiki/enwiki-feb-doc-split/*.dev' \
                --num_steps 900000 --max_seq_length 128 --lr 1e-4 --warmup_ratio 0.01 \
                --batch_size 4096  --max_predictions_per_seq 20 \
                --use_avg_len --raw --log_interval 100 \
                --ckpt_dir ./np32_ckpt_dir \
                --ckpt_interval 1000 2>&1
```

## Results

|Instance	|F1 Score	|Throughput	|Train Time	|
|---	|---	|---	|---	|
|32 x p3dn.24xlg	|NC	|5300	|853	|
|4 x p3dn.24xlg	|	|	|	|
|	|	|	|	|
|	|	|	|	|
|	|	|	|	|


