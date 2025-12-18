import json
import numpy as np
import os
import random
import torch


class WanMotionTensorDatasetWiRefMotion(torch.utils.data.Dataset):
    def __init__(self, train_json_file_list, test_json_file_list, motion_mean_path, motion_std_path, min_motion_length, max_motion_length, null_context_path, uncond_prob=0.5, is_test=False, use_global_orient=True, text_key='video_text_annot', duplicate_meta=None, t2m_prob_low_quality=0.4, t2m_prob_default=0.8, **kwargs):
        self.data_list = self._load_data_list(
            train_json_file_list=train_json_file_list,
            test_json_file_list=test_json_file_list,
            duplicate_meta=duplicate_meta,
            is_test=is_test,
        )
        print(len(self.data_list), "tensors cached in metadata.")

        motion_mean = np.load(motion_mean_path)
        motion_std = np.load(motion_std_path)

        self.motion_mean = torch.from_numpy(motion_mean).float()
        self.motion_std = torch.from_numpy(motion_std).float()
        self.motion_dim = motion_mean.shape[-1]
        self.min_motion_length = min_motion_length
        self.max_motion_length = max_motion_length
        self.uncond_prob = uncond_prob
        self.null_context = torch.load(null_context_path, weights_only=True, map_location="cpu")   # [226, 4096]
        self.is_test = is_test
        self.use_global_orient = use_global_orient
        self.text_key = text_key
        self.prompt_emb_key = f'{text_key}_wanvideot5_embed_path'
        # motion31/mogen_db captions are noisier, so use lower T2M probability by default.
        self.t2m_prob_low_quality = t2m_prob_low_quality
        # Other datasets default to a higher T2M probability.
        self.t2m_prob_default = t2m_prob_default
        total_samples = len(self.data_list)
        missing_prompt = 0
        for sample in self.data_list:
            prompt_path = sample.get(self.prompt_emb_key)
            if prompt_path is None or not os.path.exists(prompt_path):
                missing_prompt += 1
        print(
            f'[{self.__class__.__name__}] prompt embeddings with key "{self.prompt_emb_key}": '
            f'{total_samples - missing_prompt}/{total_samples} available'
        )

    def __getitem__(self, index):
        sample_index, data, motion, motion_path = self._fetch_motion_sample(index)
        motion_length = motion.shape[0]

        normalized_motion = self._normalize_motion_tensor(motion)

        data_dict = {
            "motion": normalized_motion,
            "motion_length": motion_length,
        }
        motion_dim_mask = torch.ones(self.motion_dim)

        ref_motion_dict, motion_dim_mask, attend_to_text_mask = self._prepare_ref_motion(
            data=data,
            motion=normalized_motion,
            motion_dim_mask=motion_dim_mask,
            motion_path=motion_path,
        )
        data_dict.update(ref_motion_dict)

        prompt_emb = self._resolve_prompt_embedding(data)
        prompt_emb = self._pad_prompt_embedding(prompt_emb)
        data_dict["prompt_emb"] = prompt_emb
        data_dict["prompt_length"] = prompt_emb.shape[0]
        data_dict["prompt_emb_null"] = self.null_context
        data_dict["motion_mean"] = self.motion_mean
        data_dict["motion_std"] = self.motion_std
        data_dict["text"] = data["short_annot"] if "short_annot" in data else "None"
        data_dict["motion_dim_mask"] = motion_dim_mask
        data_dict["attend_to_text_mask"] = attend_to_text_mask
        data_dict["test_sample_id"] = self._resolve_sample_id(data, sample_index)

        return data_dict

    def __len__(self):
        return len(self.data_list)

    def _load_data_list(self, train_json_file_list, test_json_file_list, duplicate_meta, is_test):
        base_json_file_list = test_json_file_list if is_test else train_json_file_list
        if duplicate_meta is not None:
            duplicate_meta = [1] if is_test else duplicate_meta
        else:
            duplicate_meta = [1] * len(base_json_file_list)

        data_list = []
        for json_file, duplicate in zip(base_json_file_list, duplicate_meta):
            with open(json_file, 'r') as f:
                file_data = json.load(f)
            data_list.extend(file_data * duplicate)
        return data_list

    def _fetch_motion_sample(self, index):
        current_index = index
        while True:
            data = self.data_list[current_index]
            motion_path = self._select_motion_path(data)
            motion = self._load_motion_tensor(motion_path)
            motion_length = motion.shape[0]
            if self.min_motion_length <= motion_length <= self.max_motion_length:
                return current_index, data, motion, motion_path
            current_index = np.random.randint(len(self.data_list))

    def _select_motion_path(self, data):
        motion_path = data["motion_path"]
        if isinstance(motion_path, list):
            motion_path = random.choice(motion_path)
        if self.is_test and not os.path.exists(motion_path):
            motion_path = './data_samples/dummy_motion.pt'
        return motion_path

    def _load_motion_tensor(self, motion_path):
        if motion_path.endswith(".pt"):
            motion = torch.load(motion_path, weights_only=True, map_location="cpu")
        elif motion_path.endswith(".npy"):
            motion = torch.from_numpy(np.load(motion_path)).float()
        else:
            raise ValueError(f"Unsupported motion file format: {motion_path}")
        if isinstance(motion, dict):
            motion = motion["motion"]
        return motion

    def _normalize_motion_tensor(self, motion):
        motion = motion[:, :self.motion_dim]
        return (motion - self.motion_mean) / self.motion_std

    def _prepare_ref_motion(self, data, motion, motion_dim_mask, motion_path):
        joint_num = 22
        if 'ref_motion_path' in data:
            ref_motion_path = data["ref_motion_path"]
            ref_motion = torch.load(ref_motion_path, weights_only=True, map_location="cpu")["motion"]
            ref_motion = ref_motion[:, :self.motion_dim]
            ref_motion = (ref_motion - self.motion_mean) / self.motion_std
        else:
            ref_motion = motion.clone()

        ref_motion_local = ref_motion[:, :(joint_num-1)*6]
        motion_mean, motion_std = self.motion_mean, self.motion_std
        ref_motion_std = motion_std[:(joint_num-1)*6]
        ref_motion_mean = motion_mean[:(joint_num-1)*6]
        if self.use_global_orient:
            ref_motion_global_orient = ref_motion[:, joint_num*12-6:joint_num*12+6]
            ref_motion_local = torch.cat([ref_motion_local, ref_motion_global_orient], dim=1)
            ref_motion_std = torch.cat([ref_motion_std, motion_std[joint_num*12-6:joint_num*12+6]])
            ref_motion_mean = torch.cat([ref_motion_mean, motion_mean[joint_num*12-6:joint_num*12+6]])

        ref_motion_local, attend_to_text_mask = self._maybe_disable_ref_motion(
            ref_motion_local, motion_path
        )
        motion_dim_mask = self._maybe_mask_motion_dim(motion_dim_mask, motion_path, joint_num, attend_to_text_mask)

        ref_motion_dict = {
            "ref_motion_original": ref_motion,
            "ref_motion": ref_motion_local.contiguous(),
            "ref_motion_mean": ref_motion_mean,
            "ref_motion_std": ref_motion_std,
            "ref_motion_length": ref_motion.shape[0],
        }
        return ref_motion_dict, motion_dim_mask, attend_to_text_mask

    def _maybe_disable_ref_motion(self, ref_motion_local, motion_path):
        attend_to_text_mask = False
        if not self.is_test:
            is_low_quality = ('motion31' in motion_path or 'mogen_db' in motion_path)
            drop_prob = self.t2m_prob_low_quality if is_low_quality else self.t2m_prob_default
            if random.random() < drop_prob:
                ref_motion_local = torch.zeros_like(ref_motion_local)    # not attend to ref motion
                attend_to_text_mask = True
        return ref_motion_local, attend_to_text_mask

    def _maybe_mask_motion_dim(self, motion_dim_mask, motion_path, joint_num, attend_to_text_mask):
        if ('mogen_db' in motion_path or 'vigen' in motion_path) and attend_to_text_mask:
            motion_dim_mask[(joint_num-1)*6:] = 0  # only supervise the local motion for t2m branch
        return motion_dim_mask

    def _resolve_prompt_embedding(self, data):
        if random.random() < self.uncond_prob and not self.is_test:
            return self.null_context

        if self.prompt_emb_key in data and os.path.exists(data[self.prompt_emb_key]):
            prompt_embed_path = data[self.prompt_emb_key]
            return torch.load(prompt_embed_path, weights_only=True, map_location="cpu")

        if self.is_test:
            raise ValueError(f'{self.prompt_emb_key} not found in data, motion_path: {data["motion_path"]}')
        return self.null_context

    def _pad_prompt_embedding(self, prompt_emb):
        if prompt_emb.shape[0] < 226:
            pad = torch.zeros(226 - prompt_emb.shape[0], prompt_emb.shape[1])
            prompt_emb = torch.cat([prompt_emb, pad], dim=0)
        return prompt_emb

    def _resolve_sample_id(self, data, index):
        if 'sample_id' in data:
            return data['sample_id']
        if 'test_sample_id' in data:
            return data['test_sample_id']
        return str(index)


class WanMotionTensorDatasetWiRefMotionM2M(WanMotionTensorDatasetWiRefMotion):
    """
    Dataset variant for motion-to-motion refinement where text embeddings are unused.
    Reuses all logic from WanMotionTensorDatasetWiRefMotion but always feeds the
    null text context instead of loading precomputed embeddings.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _resolve_prompt_embedding(self, data):
        return self.null_context.clone()


class MBenchWiRefMotion(torch.utils.data.Dataset):
    def __init__(self, test_json_file_list, motion_mean_path, motion_std_path, null_context_path, use_global_orient=True, text_key='prompt_video_detailed', test_seq_len=100, **kwargs):
        base_json_file_list = test_json_file_list
        self.data_list = []
        for json_file in base_json_file_list:
            data_list = json.load(open(json_file, 'r'))
            self.data_list.extend(data_list)

        print(len(self.data_list), "tensors cached in metadata.")
        motion_mean = np.load(motion_mean_path)
        motion_std = np.load(motion_std_path)
        self.motion_mean = torch.from_numpy(motion_mean).float()
        self.motion_std = torch.from_numpy(motion_std).float()
        self.null_context = torch.load(null_context_path, weights_only=True, map_location="cpu")   # [226, 4096]
        self.motion_dim = motion_mean.shape[-1]
        self.use_global_orient = use_global_orient
        text_key_mapping = {
            'video_text_annot': 'prompt_video_detailed',
            'motion_text_annot': 'prompt_motion_detailed'
        }
        text_key = text_key_mapping.get(text_key, text_key)
        self.text_key = text_key
        self.prompt_emb_key = f'{text_key}_wanvideot5_embed_path'
        self.test_seq_len = test_seq_len
        
    def __getitem__(self, index):
        data = self.data_list[index]
        enable_m2m = False
        if "motion_path" in data and data.get("use_ref_motion", False):
            motion_path = data["motion_path"]
            motion = torch.load(motion_path, weights_only=True, map_location="cpu")
            if isinstance(motion, dict):
                motion = motion["motion"]
            # interpolate motion from 16 fps to 20 fps
            original_frames = motion.shape[0]
            target_frames = original_frames * 20 // 16
            # [original_frames, motion_dim] -> [target_frames, motion_dim]
            motion = torch.nn.functional.interpolate(motion.unsqueeze(0).permute(0, 2, 1), size=target_frames, mode='linear', align_corners=True).squeeze(0).permute(1, 0)
            enable_m2m = True
        else:
            motion = torch.zeros((self.test_seq_len, self.motion_dim)).float()
            
        motion_length = motion.shape[0]
        motion_duration = motion_length / 20.0  # assuming 20 fps
        
        # normalize motion
        data_dict = {}
        motion = motion[:, :self.motion_dim]
        motion = (motion - self.motion_mean) / self.motion_std
        data_dict["motion"] = motion
        data_dict["motion_length"] = motion_length
        motion_dim_mask = torch.ones(self.motion_dim)

        joint_num = 22
        # ref motion
        ref_motion = motion.clone()

        # only use the local ref motion
        ref_motion_local = ref_motion[:, :(joint_num-1)*6]
        motion_mean, motion_std = self.motion_mean, self.motion_std
        ref_motion_std = motion_std[:(joint_num-1)*6]
        ref_motion_mean = motion_mean[:(joint_num-1)*6]
        if self.use_global_orient:
            ref_motion_global_orient = ref_motion[:, joint_num*12-6:joint_num*12+6]
            ref_motion_local = torch.cat([ref_motion_local, ref_motion_global_orient], dim=1)
            ref_motion_std = torch.cat([ref_motion_std, motion_std[joint_num*12-6:joint_num*12+6]])
            ref_motion_mean = torch.cat([ref_motion_mean, motion_mean[joint_num*12-6:joint_num*12+6]])
        data_dict["ref_motion_original"] = ref_motion
        data_dict["ref_motion"] = ref_motion_local.contiguous()
        data_dict["ref_motion_mean"] = ref_motion_mean
        data_dict["ref_motion_std"] = ref_motion_std
        data_dict['ref_motion_length'] = ref_motion.shape[0]

        if self.prompt_emb_key in data and os.path.exists(data[self.prompt_emb_key]):
            prompt_embed_path = data[self.prompt_emb_key]
            prompt_embed = torch.load(prompt_embed_path, weights_only=True, map_location="cpu")
            data_dict['prompt_emb'] = prompt_embed
        else:
            raise ValueError(f'{self.prompt_emb_key} not found in data, motion_path: {data["motion_path"]}')

        # pad prompt_emb to 226
        if data_dict['prompt_emb'].shape[0] < 226:
            pad = torch.zeros(226 - data_dict['prompt_emb'].shape[0], data_dict['prompt_emb'].shape[1])
            data_dict['prompt_emb'] = torch.cat([data_dict['prompt_emb'], pad], dim=0)

        prompt_length = data_dict['prompt_emb'].shape[0]
        data_dict['prompt_length'] = prompt_length
        data_dict['prompt_emb_null'] = self.null_context
        data_dict['motion_mean'] = self.motion_mean
        data_dict['motion_std'] = self.motion_std
        data_dict['text'] = data["prompt"] + f"; motion_duration: {motion_duration} seconds"
        data_dict['motion_dim_mask'] = motion_dim_mask
        data_dict['attend_to_text_mask'] = not enable_m2m
        data_dict['test_sample_id'] = data.get('global_id', str(index))

        return data_dict
    
    def __len__(self):
        return len(self.data_list)