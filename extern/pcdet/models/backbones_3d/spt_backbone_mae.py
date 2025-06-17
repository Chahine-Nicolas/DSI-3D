import numpy as np
import torch
import torch.nn as nn
from ...utils.spconv_utils import spconv
from .spt_backbone import SSTBlockV1
from ...utils import common_utils
from ...ops.sst_ops import sst_ops_utils
#from pytorch3d.loss import chamfer_distance
from typing import Union

import sys
#sys.path.insert(1, '/gpfswork/rech/xhk/ufm44cu/code/these_place_reco/GD-MAE/tools/visual_utils/')
#from matplotlib_vis import draw_scenes

# def _handle_pointcloud_input(
#     points: Union[torch.Tensor, None],
#     lengths: Union[torch.Tensor, None],
#     normals: Union[torch.Tensor, None],
# ):
#     """
#     If points is an instance of Pointclouds, retrieve the padded points tensor
#     along with the number of points per batch and the padded normals.
#     Otherwise, return the input points (and normals) with the number of points per cloud
#     set to the size of the second dimension of `points`.
#     """
#     # if isinstance(points, Pointclouds):
#     #     X = points.points_padded()
#     #     lengths = points.num_points_per_cloud()
#     #     normals = points.normals_padded()  # either a tensor or None
#     if torch.is_tensor(points):
#         if points.ndim != 3:
#             raise ValueError("Expected points to be of shape (N, P, D)")
#         X = points
#         if lengths is not None:
#             if lengths.ndim != 1 or lengths.shape[0] != X.shape[0]:
#                 raise ValueError("Expected lengths to be of shape (N,)")
#             if lengths.max() > X.shape[1]:
#                 raise ValueError("A length value was too long")
#         if lengths is None:
#             lengths = torch.full(
#                 (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
#             )
#         if normals is not None and normals.ndim != 3:
#             raise ValueError("Expected normals to be of shape (N, P, 3")
#     else:
#         raise ValueError(
#             "The input pointclouds should be either "
#             + "Pointclouds objects or torch.Tensor of shape "
#             + "(minibatch, num_points, 3)."
#         )
#     return X, lengths, normals


# def _chamfer_distance_single_direction(
#     x,
#     y,
#     x_lengths,
#     y_lengths,
#     x_normals,
#     y_normals,
#     weights,
#     batch_reduction: Union[str, None],
#     point_reduction: str,
#     norm: int,
#     abs_cosine: bool,
# ):
#     return_normals = x_normals is not None and y_normals is not None

#     N, P1, D = x.shape

#     # Check if inputs are heterogeneous and create a lengths mask.
#     is_x_heterogeneous = (x_lengths != P1).any()
#     x_mask = (
#         torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
#     )  # shape [N, P1]
#     if y.shape[0] != N or y.shape[2] != D:
#         raise ValueError("y does not have the correct shape.")
#     if weights is not None:
#         if weights.size(0) != N:
#             raise ValueError("weights must be of shape (N,).")
#         if not (weights >= 0).all():
#             raise ValueError("weights cannot be negative.")
#         if weights.sum() == 0.0:
#             weights = weights.view(N, 1)
#             if batch_reduction in ["mean", "sum"]:
#                 return (
#                     (x.sum((1, 2)) * weights).sum() * 0.0,
#                     (x.sum((1, 2)) * weights).sum() * 0.0,
#                 )
#             return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

#     cham_norm_x = x.new_zeros(())

#     x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, norm=norm, K=1)
#     cham_x = x_nn.dists[..., 0]  # (N, P1)

#     if is_x_heterogeneous:
#         cham_x[x_mask] = 0.0

#     if weights is not None:
#         cham_x *= weights.view(N, 1)

#     if return_normals:
#         # Gather the normals using the indices and keep only value for k=0
#         x_normals_near = knn_gather(y_normals, x_nn.idx, y_lengths)[..., 0, :]

#         cosine_sim = F.cosine_similarity(x_normals, x_normals_near, dim=2, eps=1e-6)
#         # If abs_cosine, ignore orientation and take the absolute value of the cosine sim.
#         cham_norm_x = 1 - (torch.abs(cosine_sim) if abs_cosine else cosine_sim)

#         if is_x_heterogeneous:
#             cham_norm_x[x_mask] = 0.0

#         if weights is not None:
#             cham_norm_x *= weights.view(N, 1)
#         cham_norm_x = cham_norm_x.sum(1)  # (N,)

#     # Apply point reduction
#     cham_x = cham_x.sum(1)  # (N,)
#     if point_reduction == "mean":
#         x_lengths_clamped = x_lengths.clamp(min=1)
#         cham_x /= x_lengths_clamped
#         if return_normals:
#             cham_norm_x /= x_lengths_clamped

#     if batch_reduction is not None:
#         # batch_reduction == "sum"
#         cham_x = cham_x.sum()
#         if return_normals:
#             cham_norm_x = cham_norm_x.sum()
#         if batch_reduction == "mean":
#             div = weights.sum() if weights is not None else max(N, 1)
#             cham_x /= div
#             if return_normals:
#                 cham_norm_x /= div

#     cham_dist = cham_x
#     cham_normals = cham_norm_x if return_normals else None
#     return cham_dist, cham_normals

# def chamfer_distance(
#     x,
#     y,
#     x_lengths=None,
#     y_lengths=None,
#     x_normals=None,
#     y_normals=None,
#     weights=None,
#     batch_reduction: Union[str, None] = "mean",
#     point_reduction: str = "mean",
#     norm: int = 2,
#     single_directional: bool = False,
#     abs_cosine: bool = True,
# ):
#     """
#     Chamfer distance between two pointclouds x and y.

#     Args:
#         x: FloatTensor of shape (N, P1, D) or a Pointclouds object representing
#             a batch of point clouds with at most P1 points in each batch element,
#             batch size N and feature dimension D.
#         y: FloatTensor of shape (N, P2, D) or a Pointclouds object representing
#             a batch of point clouds with at most P2 points in each batch element,
#             batch size N and feature dimension D.
#         x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
#             cloud in x.
#         y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
#             cloud in y.
#         x_normals: Optional FloatTensor of shape (N, P1, D).
#         y_normals: Optional FloatTensor of shape (N, P2, D).
#         weights: Optional FloatTensor of shape (N,) giving weights for
#             batch elements for reduction operation.
#         batch_reduction: Reduction operation to apply for the loss across the
#             batch, can be one of ["mean", "sum"] or None.
#         point_reduction: Reduction operation to apply for the loss across the
#             points, can be one of ["mean", "sum"].
#         norm: int indicates the norm used for the distance. Supports 1 for L1 and 2 for L2.
#         single_directional: If False (default), loss comes from both the distance between
#             each point in x and its nearest neighbor in y and each point in y and its nearest
#             neighbor in x. If True, loss is the distance between each point in x and its
#             nearest neighbor in y.
#         abs_cosine: If False, loss_normals is from one minus the cosine similarity.
#             If True (default), loss_normals is from one minus the absolute value of the
#             cosine similarity, which means that exactly opposite normals are considered
#             equivalent to exactly matching normals, i.e. sign does not matter.

#     Returns:
#         2-element tuple containing

#         - **loss**: Tensor giving the reduced distance between the pointclouds
#           in x and the pointclouds in y.
#         - **loss_normals**: Tensor giving the reduced cosine distance of normals
#           between pointclouds in x and pointclouds in y. Returns None if
#           x_normals and y_normals are None.

#     """
#     #_validate_chamfer_reduction_inputs(batch_reduction, point_reduction)

#     if not ((norm == 1) or (norm == 2)):
#         raise ValueError("Support for 1 or 2 norm.")
#     x, x_lengths, x_normals = _handle_pointcloud_input(x, x_lengths, x_normals)
#     y, y_lengths, y_normals = _handle_pointcloud_input(y, y_lengths, y_normals)

#     cham_x, cham_norm_x = _chamfer_distance_single_direction(
#         x,
#         y,
#         x_lengths,
#         y_lengths,
#         x_normals,
#         y_normals,
#         weights,
#         batch_reduction,
#         point_reduction,
#         norm,
#         abs_cosine,
#     )
#     if single_directional:
#         return cham_x, cham_norm_x
#     else:
#         cham_y, cham_norm_y = _chamfer_distance_single_direction(
#             y,
#             x,
#             y_lengths,
#             x_lengths,
#             y_normals,
#             x_normals,
#             weights,
#             batch_reduction,
#             point_reduction,
#             norm,
#             abs_cosine,
#         )
#         return (
#             cham_x + cham_y,
#             (cham_norm_x + cham_norm_y) if cham_norm_x is not None else None,
#         )

class SPTBackboneMAE(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.grid_size = grid_size
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        self.sparse_shape = grid_size[[1, 0]]
        in_channels = input_channels

        self.mask_cfg = self.model_cfg.get('MASK_CONFIG', None)
        self.mask_ratio = self.mask_cfg.RATIO if self.mask_cfg is not None else 0.0

        sst_block_list = model_cfg.SST_BLOCK_LIST
        self.sst_blocks = nn.ModuleList()
        for sst_block_cfg in sst_block_list:
            self.sst_blocks.append(SSTBlockV1(sst_block_cfg, in_channels, sst_block_cfg.NAME))
            in_channels = sst_block_cfg.ENCODER.D_MODEL
        
        in_channels = 0
        self.decoder_deblocks = nn.ModuleList()
        for src in model_cfg.FEATURES_SOURCE:
            conv_cfg = model_cfg.FUSE_LAYER[src]
            self.decoder_deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(
                    conv_cfg.NUM_FILTER, conv_cfg.NUM_UPSAMPLE_FILTER,
                    conv_cfg.UPSAMPLE_STRIDE,
                    stride=conv_cfg.UPSAMPLE_STRIDE, bias=False
                ),
                nn.BatchNorm2d(conv_cfg.NUM_UPSAMPLE_FILTER, eps=1e-3, momentum=0.01),
                nn.ReLU(inplace=True)
            ))
            in_channels += conv_cfg.NUM_UPSAMPLE_FILTER
        
        self.decoder_conv_out = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // len(self.decoder_deblocks), 3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels // len(self.decoder_deblocks), eps=1e-3, momentum=0.01),
            nn.ReLU(inplace=True)
        )
        in_channels = in_channels // len(self.decoder_deblocks)

        self.decoder_pred = nn.Linear(in_channels, self.mask_cfg.NUM_PRD_POINTS * 3, bias=True)
        self.forward_ret_dict = {}
        self.num_point_features = in_channels

    def target_assigner(self, batch_dict):
        voxel_features = batch_dict['voxel_features']
        voxel_coords = batch_dict['voxel_coords']
        voxel_shuffle_inds = batch_dict['voxel_shuffle_inds']
        points = batch_dict['points']
        point_inverse_indices = batch_dict['point_inverse_indices']
        voxel_mae_mask = batch_dict['voxel_mae_mask']
        # road_plane = batch_dict['road_plane']
        batch_size = batch_dict['batch_size']

        gt_points = sst_ops_utils.group_inner_inds(points[:, 1:4], point_inverse_indices, self.mask_cfg.NUM_GT_POINTS)
        gt_points = gt_points[voxel_shuffle_inds]
        voxel_centers = common_utils.get_voxel_centers(
            voxel_coords[:, 1:], 1, self.voxel_size, self.point_cloud_range, dim=3
        )  # (N, 3)
        norm_gt_points = gt_points - voxel_centers.unsqueeze(1)
        mask = voxel_mae_mask[voxel_shuffle_inds]
        pred_points = self.decoder_pred(voxel_features).view(voxel_features.shape[0], -1, 3)
        forward_ret_dict = {
            'pred_points': pred_points,  # (N, P1, 3)
            'gt_points': norm_gt_points,  # (N, P2, 3)
            'mask': mask  # (N,)
        }


        #print("{" + "\n".join("{!r}: {!r},".format(k, v) for k, v in batch_dict.items()) + "}")
        #ff1 = draw_scenes(points,pred_points,voxel_centers,figname="./out/" +  str(batch_dict["frame_id"][0]) +".png")
        return forward_ret_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict
        # (N, K, 3)
        gt_points, pred_points, mask = \
            self.forward_ret_dict['gt_points'], self.forward_ret_dict['pred_points'], self.forward_ret_dict['mask']
        loss, _ = (None,None) #chamfer_distance(pred_points, gt_points, weights=mask)


        return loss, tb_dict

    def forward(self, batch_dict):
        all_voxel_features, all_voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        assert torch.all(all_voxel_coords[:, 1] == 0)

        voxel_mae_mask = []
        for bs_idx in range(batch_size):
            voxel_mae_mask.append(common_utils.random_masking(1, (all_voxel_coords[:, 0] == bs_idx).sum().item(), self.mask_ratio, all_voxel_coords.device)[0])
        voxel_mae_mask = torch.cat(voxel_mae_mask, dim=0)
        batch_dict['voxel_mae_mask'] = voxel_mae_mask

        input_sp_tensor = spconv.SparseConvTensor(
            features=all_voxel_features[voxel_mae_mask == 0],
            indices=all_voxel_coords[voxel_mae_mask == 0][:, [0, 2, 3]].contiguous().int(),  # (bs_idx, y_idx, x_idx)
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = input_sp_tensor
        x_hidden = []
        
        for sst_block in self.sst_blocks:
            x = sst_block(x)
            x_hidden.append(x)

        batch_dict.update({
            'encoded_spconv_tensor': x_hidden[-1],
            'encoded_spconv_tensor_stride': self.sparse_shape[0] // x_hidden[-1].spatial_shape[0]
        })

        multi_scale_3d_features, multi_scale_3d_strides = {}, {}
        for i in range(len(x_hidden)):
            multi_scale_3d_features[f'x_conv{i + 1}'] = x_hidden[i]
            multi_scale_3d_strides[f'x_conv{i + 1}'] = self.sparse_shape[0] // x_hidden[i].spatial_shape[0]


        spatial_features = []
        spatial_features_stride = []
        for i, src in enumerate(self.model_cfg.FEATURES_SOURCE):
            per_features = multi_scale_3d_features[src].dense()
            B, Y, X = per_features.shape[0], per_features.shape[-2], per_features.shape[-1]
            spatial_features.append(self.decoder_deblocks[i](per_features.view(B, -1, Y, X)))
            spatial_features_stride.append(multi_scale_3d_strides[src] // self.model_cfg.FUSE_LAYER[src].UPSAMPLE_STRIDE)
        spatial_features = self.decoder_conv_out(torch.cat(spatial_features, dim=1))  # (B, C, Y, X)
        spatial_features_stride = spatial_features_stride[0]
        
        batch_dict['multi_scale_3d_features'] = multi_scale_3d_features
        batch_dict['multi_scale_3d_strides'] = multi_scale_3d_strides
        batch_dict['spatial_features'] = spatial_features
        batch_dict['spatial_features_stride'] = spatial_features_stride
        
        assert spatial_features.shape[0] == batch_size and spatial_features.shape[2] == self.grid_size[1] and spatial_features.shape[3] == self.grid_size[0]
        all_voxel_shuffle_inds = torch.arange(all_voxel_coords.shape[0], device=all_voxel_coords.device, dtype=torch.long)
        slices = [all_voxel_coords[:, i].long() for i in [0, 2, 3]]
        all_pyramid_voxel_features = spatial_features.permute(0, 2, 3, 1)[slices]

        target_dict = {
            'voxel_features': all_pyramid_voxel_features,
            'voxel_coords': all_voxel_coords,
            'voxel_shuffle_inds': all_voxel_shuffle_inds
        }
        batch_dict.update(target_dict)
        self.forward_ret_dict = self.target_assigner(batch_dict)

        return batch_dict
