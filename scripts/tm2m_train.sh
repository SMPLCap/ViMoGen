#!/bin/bash
python -m torch.distributed.run \
    --nproc_per_node 8 \
    --master_port 29549 \
    train_eval_vimogen.py --config ./configs/tm2m_train.yaml