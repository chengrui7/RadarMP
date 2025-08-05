import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.cuda.amp import custom_bwd, custom_fwd
from torch.autograd.function import Function, once_differentiable

from Lib.deform_attn_3d import deform3dattn_custom_cn

class Motion3DPerception(nn.Module):
    def __init__(self, value_dims_list, output_dims=32, embed_dims=128, num_heads=4, num_levels=3, num_points=150, im2col_step=64, dropout=0.1):
        super(Motion3DPerception, self).__init__() 
        self.num_heads = num_heads
        self.num_levels = num_levels
        self.num_points = num_points
        self.sampling_offset_gen = nn.Linear(embed_dims, self.num_heads * self.num_levels * self.num_points * 3)
        self.attention_weight_gen = nn.Linear(embed_dims, self.num_heads * self.num_levels * self.num_points)
        self.output_proj = nn.Linear(embed_dims, output_dims)
        self.dropout = nn.Dropout(dropout)
        self.im2col_step = im2col_step
        self.value_dims_list = value_dims_list
        self.query_projs = nn.ModuleList()
        self.value_projs = nn.ModuleList()
        for i in range(self.num_levels):
            query_proj = nn.Linear(self.value_dims_list[i], embed_dims)
            self.query_projs.append(query_proj)
            value_proj = nn.Linear(self.value_dims_list[i], embed_dims)
            self.value_projs.append(value_proj)

    def forward(self, x1_msf, x2_msf, flow_cube_layers):
        # preprocess query and value
        point_flatten = []
        query_flatten = []
        value_flatten = []
        spatial_shapes = []
        for x1f, x2f, flow_cube_layer, query_proj, value_proj in zip(x1_msf, x2_msf, flow_cube_layers, self.query_projs, self.value_projs):
            device = x1f.device
            bs, cvalue, ranl, azil, elel = x1f.shape
            spatial_shapes.append([ranl, azil, elel])
            # flatten: [B, C, D*H*W] → permute to [B, D*H*W, C]
            flow_cube_layer[..., 0] = flow_cube_layer[..., 0] / ranl  # Z 
            flow_cube_layer[..., 1] = flow_cube_layer[..., 1] / azil  # Y 
            flow_cube_layer[..., 2] = flow_cube_layer[..., 2] / elel # X
            flow_cube_layer = flow_cube_layer.view(bs, -1, ranl*azil*elel).permute(0, 2, 1).contiguous() # B N 3
            grid_r = (torch.arange(ranl, device=device) + 0.5) / ranl
            grid_a = (torch.arange(azil, device=device) + 0.5) / azil
            grid_e = (torch.arange(elel, device=device) + 0.5) / elel
            grid = torch.stack(torch.meshgrid(grid_r, grid_a, grid_e, indexing="ij"), dim=-1)  # [R, A, E, 3]
            grid = grid.view(-1, 3)  # [N, 3]
            point = grid[None,:,:] + flow_cube_layer  # [B, N, 3]
            x1f = x1f.view(bs, cvalue, ranl*azil*elel).permute(0, 2, 1).contiguous()
            x1f = query_proj(x1f)
            x2f = x2f.view(bs, cvalue, ranl*azil*elel).permute(0, 2, 1).contiguous()
            x2f = value_proj(x2f)
            point_flatten.append(point)
            query_flatten.append(x1f)
            value_flatten.append(x2f)
        reference_points = torch.cat(point_flatten, dim=1).contiguous()
        x1_query = torch.cat(query_flatten, dim=1).contiguous()
        x2_value = torch.cat(value_flatten, dim=1).contiguous()
        device = x1_query.device
        value_spatial_shapes = torch.tensor(spatial_shapes, dtype=torch.long, device=device) # N * 3
        level_start_index = [0]
        for shape in spatial_shapes:
            d, h, w = shape
            level_start_index.append(level_start_index[-1] + d * h * w)
        value_level_start_index = torch.tensor(level_start_index[:-1], dtype=torch.long, device=device) # N * 1
        # pre Query Value
        bs, num_query, embed_dims = x1_query.shape
        query = x1_query
        bs, num_value, embed_dims = x2_value.shape
        value = x2_value.view(bs, num_value, self.num_heads, embed_dims//self.num_heads)
        # attention weight
        attention_weights = self.attention_weight_gen(x1_query)
        attention_weights = attention_weights.view(bs, num_query,  self.num_heads, self.num_levels * self.num_points)
        attention_weights = F.softmax(attention_weights, dim=-1)
        attention_weights = attention_weights.reshape(bs, num_query, self.num_heads, self.num_levels, self.num_points)
        attention_weights = attention_weights.contiguous()
        # offsets
        sampling_offsets = self.sampling_offset_gen(x1_query)
        sampling_offsets = sampling_offsets.view(bs, num_query, self.num_heads, self.num_levels, self.num_points, 3)
        sampling_offsets = sampling_offsets.reshape(bs, num_query, self.num_heads, self.num_levels, self.num_points, 3)
        sampling_offsets = sampling_offsets / value_spatial_shapes[None, None, None, :, None, :]
        sampling_offsets = reference_points[:, :, None, None, None, :] + sampling_offsets
        sampling_offsets = sampling_offsets.contiguous()
        # deformable attention
        im2col_step = self.im2col_step
        # if device.index == 0:
        #     print('deformable shape: ')
        #     print(value.shape)
        #     print(sampling_offsets.shape)
        #     print(attention_weights.shape)
        #     print(value_spatial_shapes.shape)
        #     print(value_level_start_index.shape)
        #     print(value_spatial_shapes)
        #     print(value_level_start_index)
        output = MultiScaleDeformableAttn3DCustomFunction_fp32.apply(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_offsets,
            attention_weights, 
            im2col_step)

        output = self.output_proj(output)
        output = self.dropout(output) + query

        return output, spatial_shapes


class MultiScaleDeformableAttn3DCustomFunction_fp32(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = deform3dattn_custom_cn.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        deform3dattn_custom_cn.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None