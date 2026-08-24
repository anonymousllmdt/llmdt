import torch

from utils.net_utils import L1_loss, L2_loss, cross_entropy
from utils.registry import Registry
LOSS = Registry('loss')



@LOSS.register()
def cal_recon_trans_loss(loss, batch, loss_stats):
    coord = loss.coord
    l1_loss = cal_L1_with_mask(loss, batch[f'smplx_params_{coord}']['transl'], batch['recon_transl'], batch['motion_mask'])
    loss_stats['recon_trans'] = l1_loss.detach().cpu()
    return l1_loss

@LOSS.register()
def cal_recon_orient_6d_loss(loss, batch, loss_stats):
    l1_loss = cal_L1_with_mask(loss, batch['orient_6d'], batch['recon_orient_6d'], batch['motion_mask'])
    loss_stats['recon_orient_6d'] = l1_loss.detach().cpu()
    return l1_loss

@LOSS.register()
def cal_recon_pose_6d_loss(loss, batch, loss_stats):
    l1_loss = cal_L1_with_mask(loss, batch['body_pose_6d'], batch['recon_body_pose_6d'], batch['motion_mask'])
    loss_stats['recon_body_pose_6d'] = l1_loss.detach().cpu()
    return l1_loss


def cal_L1_with_mask(loss, x, y, mask):
    '''
    Input:
        x, y: (B, S, D)
        mask: (B, S)
    Output:
        l1_loss: (B)
    '''
    l1_loss = L1_loss(x, y).mean(-1) * (~mask)
    l1_loss = l1_loss.sum(-1) / (~mask).sum(-1)
    return l1_loss