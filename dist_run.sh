GPUS=${GPUS:-2}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29505}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

echo "Distributed parameters #########"
echo "nnodes: $NNODES"
echo "node rank: $NODE_RANK"
echo "port: $PORT"
echo "master: $MASTER_ADDR"
echo "num_gpus: $GPUS"

python3.8 -m torch.distributed.run \
    --master_addr=$MASTER_ADDR \
    --master_port=$PORT \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$GPUS \
    train.py \
    --distributed \
    --launcher pytorch_elastic $@