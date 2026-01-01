#!/bin/bash
# Extract text embeddings for ViMoGen-228K training data
# This script processes the full training dataset (228K samples)
# Note: This takes several hours on a single GPU

json_file=./data/meta_info/ViMoGen-228K_train.json
save_dir=./data/ViMoGen-228K/text_embeddings

CUDA_VISIBLE_DEVICES=0 python ./models/transformer/wan/text_encoding_batch.py --json_file $json_file --text_key 'video_text_annot' --save_dir $save_dir

echo "Text encodings for training data are done."
