import numpy as np
import os
from collections import OrderedDict
from torch.utils.data import Sampler
from typing import Any, Iterator, List, Tuple

from .bucket_config import BucketConfig


class BucketFactory:

    def __init__(
        self,
        ori_thw_list: List[Tuple],
        dp_size: int,
        rnd_state: np.random.RandomState,
        bucket_config: BucketConfig,
    ) -> None:
        self.ori_thw_list = ori_thw_list
        self.dp_size = dp_size
        self.rnd_state = rnd_state
        self.bucket_config = bucket_config

    def __call__(self, ) -> Any:
        flop_list = [
            self.bucket_config(*thw, self.rnd_state)
            for thw in self.ori_thw_list
        ]

        sorted_indices = sorted(
            range(len(flop_list)), key=lambda x: flop_list[x])

        bucket_dict = OrderedDict({})
        for i, idx in enumerate(sorted_indices):
            # video_size = flop_list[idx].size
            bucket_key = flop_list[idx]
            if bucket_key not in bucket_dict:
                bucket_dict[bucket_key] = []
            bucket_dict[bucket_key].append(idx)

        def merge_bucket(bucket_dict: OrderedDict, min_length: int):
            all_pass = False
            while not all_pass:
                all_pass = True
                keys = list(bucket_dict.keys())
                to_pop_keys = []
                for key_idx, (key, idx_list) in enumerate(bucket_dict.items()):
                    if len(idx_list) < min_length:
                        all_pass = False
                        if key_idx == len(bucket_dict) - 1:
                            continue
                        tgt_key = keys[key_idx + 1]
                        bucket_dict[tgt_key] = idx_list + bucket_dict[tgt_key]
                        to_pop_keys.append(key)
                for key in to_pop_keys:
                    bucket_dict.pop(key)
                keys = list(bucket_dict.keys())
                if len(bucket_dict[keys[-1]]) < min_length:
                    bucket_dict[keys[-2]] = bucket_dict[
                        keys[-2]] + bucket_dict[keys[-1]]
                    bucket_dict.pop(keys[-1])
            return bucket_dict

        for bucket_key, idx_list in bucket_dict.items():
            print(
                f'before merging, bucket_key: {bucket_key}, consists of {len(idx_list)} clips'
            )

        bucket_dict = merge_bucket(bucket_dict, min_length=self.dp_size)
        for bucket_key, idx_list in bucket_dict.items():
            print(
                f'after merging, bucket_key: {bucket_key}, consists of {len(idx_list)} clips'
            )
        final_idx_list = []
        final_bucket_key_list = []

        for video_size, idx_list in bucket_dict.items():
            print(
                f'before appending, bucket_key: {video_size}, consists of {len(idx_list)} clips'
            )
            if (res := (len(idx_list) % self.dp_size)) != 0:
                idx_list = idx_list + idx_list[:self.dp_size - res]
                bucket_dict[video_size] = idx_list
            print(
                f'after appending, bucket_key: {video_size}, consists of {len(idx_list)} clips'
            )
            final_idx_list += idx_list
            final_bucket_key_list += [video_size] * len(idx_list)
        return final_idx_list, flop_list, final_bucket_key_list


class DistributedFlopBalanceSampler(Sampler):

    def __init__(
        self,
        dataset,
        dp_rank: int,
        dp_size: int,
        global_seed: int = 0,
        bucket_config_type: str = 'DefaultBucketConfigNotExact',
        call_set_epoch: bool = True,
    ) -> None:
        self.dataset = dataset
        self.global_seed = global_seed
        self.dp_rank = dp_rank
        self.dp_size = dp_size

        assert bucket_config_type in [
            'DefaultBucketConfigNotExact', 'DefaultBucketConfig',
            'BucketConfigHardCoded1', 'BucketConfigHardCoded2',
            'BucketConfigHardCoded3', 'DefaultBucketConfig3ARNotExact'
        ]
        bucket_config = BucketConfig.from_class_name(bucket_config_type, {})
        ori_size_list = self.get_ori_size_list()
        # ensure the global_seed is the same across different processes
        self.rnd_state = np.random.RandomState(global_seed)
        self.bucket_factory = BucketFactory(
            ori_size_list,
            dp_size=self.dp_size,
            rnd_state=self.rnd_state,
            bucket_config=bucket_config,
        )
        if call_set_epoch:
            self.set_epoch(0)

    def get_ori_size_list(self, ):
        ori_size_list = [None] * len(self.dataset.data_list)
        for i, data in enumerate(self.dataset.data_list):
            ori_size_list[i] = (data['length'], data['height'], data['width'])
        return ori_size_list

    def bucket_prepare(self, ):
        print('----------prepare bucket-----------')
        self.final_idx_list, self.flop_list, self.final_bucket_key_list = self.bucket_factory(
        )
        assert len(self.final_idx_list) >= len(self.flop_list)
        assert len(self.final_idx_list) % self.dp_size == 0
        print('----------bucket prepared-----------')

    def set_epoch(self, epoch: int) -> None:
        """different from DistributedSampler, you don't need to call this
        function at the start of each epoch, because the return Iterator of
        __iter__ will be different for each epoch originally."""
        self.epoch_count = epoch
        self.rnd_state.seed(epoch + self.global_seed)
        self.bucket_prepare()

    def __len__(self, ):
        return len(self.final_idx_list) // self.dp_size

    def __iter__(self) -> Iterator:
        assert len(self.final_idx_list) % self.dp_size == 0
        indices = np.array(self.rnd_state.permutation(len(self)))
        # indices = self.rnd_state.choice(len(self))
        # indices = torch.randperm(len(self), generator=self.generator)
        indices = (indices * self.dp_size + self.dp_rank).tolist()
        # idx_size_list = [None] * len(indices)
        for idx in indices:
            data_idx = self.final_idx_list[idx]
            t, h, w = self.flop_list[data_idx].size
            bucket_key = self.final_bucket_key_list[idx]
            yield f'{data_idx}-{t}-{h}-{w}-{bucket_key}'
        # self.set_epoch(self.epoch_count + 1)
        # self.bucket_prepare()