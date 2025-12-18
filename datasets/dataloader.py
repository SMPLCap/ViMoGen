import collections
import numpy as np
import random
import torch
from torch.utils.data import DataLoader

from .multi_resolution_sampler import DistributedFlopBalanceSampler
from .sampler import StatefulDistributedSampler
from .video_datasets import (
    WanMotionTensorDatasetWiRefMotion,
    MBenchWiRefMotion
)


# Deterministic dataloader
def get_seed_worker(seed):

    def seed_worker(worker_id):
        worker_seed = seed
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        random.seed(worker_seed)

    return seed_worker


def get_dataloader(
    local_batch,
    dp_mesh,
    seed,
    dataset_args,
    num_workers,
    bucket_config_type: str = None,
    dataset_name: str = 'ZoeDataset',
    is_test: bool = False,
):
    drop_last = False
    collate_fn = collate_fn_default 
    if dataset_name == 'WanMotionTensorDatasetWiRefMotion':
        data_cls = WanMotionTensorDatasetWiRefMotion
        collate_fn = collate_fn_motion_wanvideo
        dataset_args['is_test'] = is_test
    elif dataset_name == 'MBenchWiRefMotion':
        data_cls = MBenchWiRefMotion
        collate_fn = collate_fn_motion_wanvideo
        assert is_test == True, "MBench Dataset is for testing only"
    else:
        data_cls = ZoeMixDataset
    dataset_args['dp_rank'] = dp_mesh.get_local_rank()
    dataset = data_cls(**dataset_args)
    if bucket_config_type is None:
        sampler = StatefulDistributedSampler(
            dataset,
            num_replicas=dp_mesh.size(),
            rank=dp_mesh.get_local_rank(),
            shuffle=not is_test,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=local_batch,
            sampler=sampler,
            worker_init_fn=get_seed_worker(seed),
            drop_last=drop_last,
            pin_memory=True,
            num_workers=num_workers,
            collate_fn=collate_fn,
            prefetch_factor=2,
        )
    else:
        sampler = DistributedFlopBalanceSampler(
            dataset,
            dp_rank=dp_mesh.get_local_rank(),
            dp_size=dp_mesh.size(),
            global_seed=seed,
            bucket_config_type=bucket_config_type,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=local_batch,
            sampler=sampler,
            drop_last=drop_last,
            pin_memory=True,
            num_workers=num_workers,
        )
    return dataloader, sampler


def collate_fn_default(batch):
    # filter out None
    batch = [x for x in batch if x is not None]

    # HACK: for loading text features
    use_mask = False
    if 'mask' in batch[0] and isinstance(batch[0]['mask'], int):
        masks = [x.pop('mask') for x in batch]

        texts = [x.pop('text') for x in batch]
        texts = torch.cat(texts, dim=1)
        use_mask = True

    ret = torch.utils.data.default_collate(batch)

    if use_mask:
        ret['mask'] = masks
        ret['text'] = texts
    return ret


def collate_fn_motion_wanvideo(batch):
    # filter out None
    batch = [x for x in batch if x is not None]

    # collate motion
    motions = [x.pop('motion') for x in batch]
    motions = collate_tensors(motions)
    motion_lengths = [x.get('motion_length') for x in batch]
    motion_mask = lengths_to_mask(motion_lengths, motions.device, motions.shape[1])

    with_ref_motion = False
    if 'ref_motion' in batch[0]:
        with_ref_motion = True
        ref_motions = [x.pop('ref_motion') for x in batch]
        ref_motions_original = [x.pop('ref_motion_original') for x in batch]
        ref_motions = collate_tensors(ref_motions)
        ref_motions_original = collate_tensors(ref_motions_original)
        ref_motion_lengths = [x.pop('ref_motion_length') for x in batch]
        ref_motion_mask = lengths_to_mask(ref_motion_lengths, ref_motions.device, ref_motions.shape[1])

    # collate prompt
    with_prompt_emb = False
    if 'prompt_emb' in batch[0]:
        with_prompt_emb = True
        prompt_emb = [x.pop('prompt_emb') for x in batch]
        prompt_emb = collate_tensors(prompt_emb)
        prompt_lengths = [x.pop('prompt_length') for x in batch]
        prompt_emb_mask = lengths_to_mask(prompt_lengths, prompt_emb.device, prompt_emb.shape[1])

    texts = None
    if 'text' in batch[0]:
        texts = [x.pop('text') for x in batch]

    sample_ids = None
    if 'test_sample_id' in batch[0]:
        sample_ids = [x.pop('test_sample_id') for x in batch]

    try:
        ret = torch.utils.data.default_collate(batch)
    except Exception as e:
        print(f'Failed to collate batch in collate_fn_motion_wanvideo: {e}')
        for idx, sample in enumerate(batch[:2]):
            print(f'sample {idx} keys: {list(sample.keys())}')
        raise
    
    ret['motion'] = motions
    ret['motion_mask'] = motion_mask
    if with_ref_motion:
        ret['ref_motion'] = ref_motions
        ret['ref_motion_mask'] = ref_motion_mask
        ret['ref_motion_original'] = ref_motions_original
    if with_prompt_emb:
        ret['prompt_emb'] = prompt_emb
        ret['prompt_emb_mask'] = prompt_emb_mask
    if texts is not None:
        ret['text'] = texts
    if sample_ids is not None:
        ret['test_sample_id'] = sample_ids
    return ret


def collate_tensors(batch: list) -> torch.Tensor:
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch), ) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


def lengths_to_mask(lengths: list[int],
                    device: torch.device,
                    max_len: int = None) -> torch.Tensor:
    lengths = torch.tensor(lengths, device=device)
    max_len = max_len if max_len else max(lengths)
    mask = torch.arange(max_len, device=device).expand(
        len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
