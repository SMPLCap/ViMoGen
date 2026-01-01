import torch
import yaml
import os
from .net_modules import StructureEncoder, DFNet, StructureDecoder

class NRDF(torch.nn.Module):
    
    def __init__(self, opt):
        super(NRDF, self).__init__()

        self.device = opt['train']['device']
        self.njoints = opt['model']['DFNet']['num_parts']
        self.enc = StructureEncoder(opt['model']['StrEnc']).to(self.device)

        self.dfnet = DFNet(opt['model']['DFNet']).to(self.device)
        self.dec = StructureDecoder(opt['model']['StrEnc']).to(self.device)

        self.exp_name = opt['experiment']['exp_name']

        self.loss = opt['train']['loss_type']
        self.batch_size = opt['train']['batch_size']

        if self.loss == 'l1':
            self.loss_l1 = torch.nn.L1Loss()
        elif self.loss == 'l2':
            self.loss_l1 = torch.nn.MSELoss()

        self.loss_l2 = torch.nn.MSELoss()

    def train(self, mode=True):
        super().train(mode)

    def forward(self, pose_in, dist_gt=None, man_poses=None, train=True, eikonal=0.0):

        if train and eikonal > 0.0:
            pose_in.requires_grad = True

        if dist_gt is not None:
            dist_gt = dist_gt.reshape(-1)
        
        pose_latent = self.enc(pose_in)

        dist_pred = self.dfnet(pose_latent)
        return {'dist_pred': dist_pred}

def load_config(path):
    """ load config file"""
    with open(path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    return cfg


def load_model(model_dir):
    checkpoint_path = os.path.join(model_dir, 'checkpoints', 'checkpoint_epoch_best.tar')
    config_file = os.path.join(model_dir, 'config.yaml')
    model = NRDF(load_config(config_file))
    checkpoint = torch.load(checkpoint_path, map_location='cpu')['model_state_dict']
    model.load_state_dict(checkpoint)
    model.eval()
    return model