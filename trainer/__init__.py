from .base_trainer import TrainerBase, linear_lr_warmpup, update_ema
from .scheduler import (
    get_cfg_scale_list,
    get_cogvideo_dynamic_cfg_scale_list,
    get_smooth_dynamic_cfg_scale_list,
)
from .sd3_scheduler import SD3RectFlow


def dummy_dataloader_for_debug(batch_size: int = 4, num_workers: int = 4):
    import torch
    from torch.utils.data import DataLoader, Dataset

    class DummyDataset(Dataset):

        def __init__(self) -> None:
            super().__init__()
            data = torch.randn(
                1,
                3,
                1024,
                1024,
            )
            self.data = data.expand(100000, -1, -1, -1)

        def __len__(self, ):
            return len(self.data)

        def __getitem__(self, index):
            return (
                self.data[index],
                self.data[index],
                'A cat holding a sign that says hello world',
            )

    dataset = DummyDataset()
    train_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True)
    val_loader = train_loader
    return train_loader, val_loader


__all__ = [
    'TrainerBase',
    'linear_lr_warmpup',
    'update_ema',
    'SD3RectFlow',
    'get_cfg_scale_list',
    'get_cogvideo_dynamic_cfg_scale_list',
    'get_smooth_dynamic_cfg_scale_list',
]
