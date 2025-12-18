#!/bin/bash
python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29549 \
    train_eval_vimogen.py --mode eval --config ./configs/t2m_infer.yaml --mbench_name smoking_test