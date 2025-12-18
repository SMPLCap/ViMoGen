#!/bin/bash

json_file=./data_samples/example_archive.json
save_dir=./data_samples/text_embeddings

CUDA_VISIBLE_DEVICES=0 python ./models/transformer/wan/text_encoding_batch.py --json_file $json_file --text_key 'prompt' --save_dir $save_dir

echo "All text encodings are done."