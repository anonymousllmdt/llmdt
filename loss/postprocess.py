import torch
from utils.geo_transform import rot_6d_to_axis_angle
from utils.registry import Registry
POST = Registry('postprocess')


@POST.register()
def get_pd_zero_vertices(loss, batch):
    zero_verts = get_zero_verts(loss, batch, batch['recon_body_pose'])
    batch.update({
        'pd_zero_verts': zero_verts,
    })
    
    
@POST.register()
def get_pd_vertices(loss, batch):
    if len(batch['recon_body_pose'].shape) == 3:
        B, S, _ = batch['smplx_params']['transl'].shape
        smplx_params = {
            'betas': batch['smplx_params']['betas'],
            'transl': batch['recon_transl'],
            'global_orient': rot_6d_to_axis_angle(batch['recon_orient_6d']),
            'body_pose': batch['recon_body_pose'], 
        }
        smplx_params = {k: v.reshape(B * S, -1) for k, v in smplx_params.items()}
        
        body_opt = loss.smplx_model(**smplx_params, return_verts=True)
        batch.update({
            'pd_verts': body_opt.vertices.reshape(B, S, -1, 3),
            'pd_joints': body_opt.joints.reshape(B, S, -1, 3),
        })
    elif len(batch['recon_body_pose'].shape) == 4:
        B, K, S, _ = batch['recon_body_pose'].shape
        smplx_params = {
            'betas': batch['smplx_params']['betas'].repeat((1, K, 1, 1)),
            'transl': batch['recon_transl'],
            'global_orient': rot_6d_to_axis_angle(batch['recon_orient_6d']),
            'body_pose': batch['recon_body_pose'], 
        }
        smplx_params = {k: v.reshape(B * K * S, -1) for k, v in smplx_params.items()}
        
        body_opt = loss.smplx_model(**smplx_params, return_verts=True)
        batch.update({
            'pd_verts': body_opt.vertices.reshape(B, K, S, -1, 3),
            'pd_joints': body_opt.joints.reshape(B, K, S, -1, 3),
        })
    
def get_zero_verts(loss, batch, localpose):
    if len(localpose.shape) == 3:
        B, S, _ = batch['smplx_params']['transl'].shape
        smplx_params = {
            'betas': batch['smplx_params']['betas'],
            'body_pose': localpose, 
        }
        smplx_params = {k: v.reshape(B * S, -1) for k, v in smplx_params.items()}
        body_opt = loss.smplx_model(**smplx_params, return_verts=True)
        body_vertices = body_opt.vertices
        return body_vertices.reshape(B, S, -1, 3)
    elif len(localpose.shape) == 4:
        B, K, S, _ = localpose.shape
        smplx_params = {
            'betas': batch['smplx_params']['betas'].repeat((1, K, 1, 1)),
            'body_pose': localpose, 
        }
        smplx_params = {k: v.reshape(B * K * S, -1) for k, v in smplx_params.items()}
        body_opt = loss.smplx_model(**smplx_params, return_verts=True)
        body_vertices = body_opt.vertices
        return body_vertices.reshape(B, K, S, -1, 3)