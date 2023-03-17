from typing import Optional, List
import math
import copy

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_

from utils.misc import inverse_sigmoid
from .ops.modules import MSDeformAttn


class DepthAwareTransformer(nn.Module):
    def __init__(
            self,
            d_model=256,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=1024,
            dropout=0.1,
            activation="relu",
            return_intermediate_dec=False,
            num_feature_levels=4,
            dec_n_points=4,
            enc_n_points=4,
            two_stage=False,
            two_stage_num_proposals=50):

        super().__init__()

        self.d_model = d_model
        self.nhead = nhead
        self.two_stage = two_stage
        self.two_stage_num_proposals = two_stage_num_proposals

        encoder_layer = VisualEncoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, enc_n_points)
        self.encoder = VisualEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = DepthAwareDecoderLayer(
            d_model, dim_feedforward, dropout, activation, num_feature_levels, nhead, dec_n_points)
        self.decoder = DepthAwareDecoder(decoder_layer, num_decoder_layers, return_intermediate_dec)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        if two_stage:
            self.enc_output = nn.Linear(d_model, d_model)
            self.enc_output_norm = nn.LayerNorm(d_model)
            self.pos_trans = nn.Linear(d_model * 2, d_model * 2)
            self.pos_trans_norm = nn.LayerNorm(d_model * 2)
        else:
            self.reference_points = nn.Linear(d_model, 2)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if not self.two_stage:
            xavier_uniform_(self.reference_points.weight.data, gain=1.0)
            constant_(self.reference_points.bias.data, 0.)
        normal_(self.level_embed)

    def get_proposal_pos_embed(self, proposals):
        num_pos_feats = 128
        temperature = 10000
        scale = 2 * math.pi

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=proposals.device)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
        # N, L, 4
        proposals = proposals.sigmoid() * scale
        # N, L, 4, 128
        pos = proposals[:, :, :, None] / dim_t
        # N, L, 4, 64, 2
        pos = torch.stack((pos[:, :, :, 0::2].sin(), pos[:, :, :, 1::2].cos()), dim=4).flatten(2)
        return pos

    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

            lr = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            tb = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            wh = torch.cat((lr, tb), -1)

            proposal = torch.cat((grid, wh), -1).view(N_, -1, 6)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(self, srcs, masks, pos_embeds, query_embed=None, depth_pos_embed=None, weighted_depth=None):
        assert self.two_stage or query_embed is not None

        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)

            mask = mask.flatten(1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)

        src_flatten = torch.cat(src_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=srcs[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)

        # encoder
        memory = self.encoder(src_flatten, spatial_shapes, level_start_index, valid_ratios, lvl_pos_embed_flatten,
                              mask_flatten)

        # prepare input for decoder
        bs, _, c = memory.shape
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flatten, spatial_shapes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_class = self.decoder.class_embed[self.decoder.num_layers](output_memory)
            enc_outputs_coord_unact = self.decoder.bbox_embed[self.decoder.num_layers](output_memory) + output_proposals

            topk = self.two_stage_num_proposals
            topk_proposals = torch.topk(enc_outputs_class[..., 0], topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 6))
            topk_coords_unact = topk_coords_unact.detach()
            reference_points = topk_coords_unact.sigmoid()
            init_reference_out = reference_points

            topk_coords_unact_input = torch.cat(
                (topk_coords_unact[..., 0: 2], topk_coords_unact[..., 2:: 2] + topk_coords_unact[..., 3:: 2]), dim=-1)

            pos_trans_out = self.pos_trans_norm(self.pos_trans(self.get_proposal_pos_embed(topk_coords_unact_input)))
            query_embed, tgt = torch.split(pos_trans_out, c, dim=2)
        else:
            learnable_anchor_box_3d = query_embed[:, :3].sigmoid()

            tgt = query_embed[:, 3:259]
            # query_embed_0 = query_embed_0.unsqueeze(0).expand(bs, -1, -1)
            tgt = tgt.unsqueeze(0).expand(bs, -1, -1).sigmoid()
            # reference_points = self.reference_points(tgt).sigmoid()  # (16,50,2)
            # reference_points = learnable_anchor_box_3d[:, 0:2]
            # init_reference_out = reference_points

        depth_pos_embed = depth_pos_embed.flatten(2).permute(2, 0, 1)  # 1920,16,256
        mask_depth = masks[1].flatten(1)

        # decoder
        hs, inter_references, inter_references_dim, d_reg = self.decoder(
            tgt,
            memory,
            spatial_shapes,
            level_start_index,
            valid_ratios,
            learnable_anchor_box_3d,
            mask_flatten,
            depth_pos_embed,
            mask_depth,
            weighted_depth,
            )

        inter_references_out = inter_references
        inter_references_out_dim = inter_references_dim

        if self.two_stage:
            return hs, inter_references_out, inter_references_out_dim, d_reg, enc_outputs_class, enc_outputs_coord_unact
        return hs, inter_references_out, inter_references_out_dim, d_reg, None, None


class VisualEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self, src, pos, reference_points, spatial_shapes, level_start_index, padding_mask=None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index,
                              padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src


class VisualEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self, src, spatial_shapes, level_start_index, valid_ratios, pos=None, padding_mask=None):
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        for _, layer in enumerate(self.layers):
            output = layer(output, pos, reference_points, spatial_shapes, level_start_index, padding_mask)

        return output


class DepthAwareDecoderLayer(nn.Module):
    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4):
        super().__init__()

        # cross attention
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # depth cross attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_sine_proj = nn.Linear(d_model, d_model)
        self.cross_attn_depth = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout_depth = nn.Dropout(dropout)
        self.norm_depth = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(self,
                tgt,
                query_pos,
                reference_points,
                src,
                src_spatial_shapes,
                level_start_index,
                src_padding_mask,
                depth_pos_embed,
                mask_depth,
                query_sine_embed
                ):
        # depth cross attention
        q = tgt.transpose(0, 1) + query_pos.transpose(0, 1)
        tgt2 = self.cross_attn_depth(q,
                                     depth_pos_embed,
                                     depth_pos_embed,
                                     key_padding_mask=mask_depth)[0].transpose(0, 1)
        tgt = tgt + self.dropout_depth(tgt2)
        tgt = self.norm_depth(tgt)

        # self attention
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q.transpose(0, 1), k.transpose(0, 1), tgt.transpose(0, 1))[0].transpose(0, 1)
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # visual cross attention


        # 第一层由于没有足够的位置信息，因此默认要加上位置部分

        # 将 query & key 的内容与通过正余弦位置编码得到的位置部分拼接，
        # 从而两者在交叉注意力中做交互是能够实现 内容与位置分别独立做交互，即：
        # q_content <-> k_content; q_position <-> k_position
        # q = q.view(50, 16, 8, 256 // 8)
        # query_sine_embed 由 4d anchor box 经历正余弦位置编码而来，实现了与 key 一致的位置编码方式
        # query_sine_embed = self.ca_qpos_sine_proj(query_sine_embed)
        # 这里有个细节要注意，就是在拼接(concat)前要先将最后一维按注意力头进行划分，这样才能将各个头部的维度对应拼接
        # 否则，就会导致前面一些头部全部都是 q 的部分，而后面一些头部则全是 query_sine_embed 的部分。
        # query_sine_embed = query_sine_embed.view(50, 16, 8, 256 // 8)
        # q = torch.cat([q, query_sine_embed], dim=3).view(50, 16, 256 * 2)
        # q = torch.cat([q, query_sine_embed], dim=2)
        # q = self.with_pos_embed(q, query_sine_embed)
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, query_sine_embed),
                               reference_points,
                               src, src_spatial_shapes, level_start_index, src_padding_mask)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class DepthAwareDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.return_intermediate = return_intermediate
        # hack implementation for iterative bounding box refinement and two-stage Deformable DETR
        self.bbox_embed = None
        self.d_refnet = MLP(256, 256, 2, 2)
        # self.depth_embed = MLP(256, 256, 2, 2)
        self.d_refnet = _get_clones(self.d_refnet, 3)

        self.dim_embed = None
        self.class_embed = None
        self.ref_point_head = MLP(3 * 128, 256, 256, 2)
        # self.ref_point_head_6 = MLP(4 // 2 * 256, 256, 256, 2)
        self.query_scale = MLP(256, 256, 256, 2)
        # self.ref_anchor_head = MLP(256, 256, 2, 2)
        # self.high_dim_query_proj = MLP(256, 256, 256, 3)
    def forward(self, tgt, src, src_spatial_shapes, src_level_start_index, src_valid_ratios,
                learnable_anchor_box_3d_unit=None, src_padding_mask=None, depth_pos_embed=None, mask_depth=None,
                weighted_depth=None):
        output = tgt
        intermediate = []
        intermediate_reference_points = []
        intermediate_reference_dims = []
        depth_reg = []
        bs = src.shape[0]
        # reference_points = reference_points[None].repeat(bs, 1, 1)
        learnable_anchor_box_3d = learnable_anchor_box_3d_unit[None].repeat(bs,1,1).sigmoid()

        # reference_points = gen_sineembed_for_position_for_ref(reference_points)
        for lid, layer in enumerate(self.layers):
            if learnable_anchor_box_3d.shape[-1] == 7:  # x,y,t,b,l,r,d
                reference_points = learnable_anchor_box_3d[..., :6]
                learnable_anchor_box_3d = torch.cat((learnable_anchor_box_3d[..., :2],learnable_anchor_box_3d[..., 6:7]),dim= -1)
                reference_points_input = reference_points[:, :, None] \
                                         * torch.cat([src_valid_ratios, src_valid_ratios, src_valid_ratios], -1)[:,
                                           None]
            else:
                # print(reference_points.shape[-1])
                assert learnable_anchor_box_3d.shape[-1] == 3
                # reference_points_input = reference_points[:, :, None] * torch.cat([src_valid_ratios, src_valid_ratios],
                #                                                                   -1)[:, None]
                reference_points = learnable_anchor_box_3d[..., :2]
                reference_points_input = reference_points[:, :, None] * src_valid_ratios[:, None]  # 16,50,4,2

            query_sine_embed = gen_sineembed_for_position(learnable_anchor_box_3d)  # 16,50,256 PE(x,y)
            query_pos = self.ref_point_head(query_sine_embed)  # 16,50,256 MLP(PE(x,y))
            if lid == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)
            query_sine_embed = query_sine_embed[..., :256] * pos_transformation
            # if lid != 0:
            #     learnable_anchor_box_3d = learnable_anchor_box_3d + self.high_dim_query_proj(output)   # x+x',y+y'16,50,256
            refD_cond = F.grid_sample(
                weighted_depth.unsqueeze(1),
                learnable_anchor_box_3d[:, :, 0:2].unsqueeze(2).detach(),
                mode='bilinear',
                align_corners=True).squeeze(1)  # nq, bs, 2
            query_sine_embed_d = query_sine_embed * (refD_cond[..., 0] / learnable_anchor_box_3d[..., 2]).unsqueeze(-1)
            # query_sine_embed[..., :256 // 2] *= (refD_cond[..., 0] / learnable_anchor_box_3d[..., 2]).unsqueeze(-1)
            output = layer(output,
                           query_pos,
                           reference_points_input,
                           src,
                           src_spatial_shapes,
                           src_level_start_index,
                           src_padding_mask,
                           depth_pos_embed,
                           mask_depth,
                           query_sine_embed_d
                           )

            # hack implementation for iterative bounding box refinement
            if self.bbox_embed is not None:
                tmp = self.bbox_embed[lid](output)   # x',y',t',b',l',r'
                d_ref = self.d_refnet[lid](output)    # d',deviation

                if learnable_anchor_box_3d.shape[-1] == 7:
                   new_learnable_anchor_box_3d = tmp + inverse_sigmoid(learnable_anchor_box_3d[..., :6])  #  x,y = x',y'+x,y
                   d_ref[..., :1] = d_ref[..., :1] + inverse_sigmoid(learnable_anchor_box_3d[..., 2:3])  # d = d'+d
                   new_learnable_anchor_box_3d = torch.cat((new_learnable_anchor_box_3d.sigmoid(), d_ref[..., :1].sigmoid()), 2)
                   # new(x,y,t,b,l,r,d)
                else:
                    # print(reference_points.shape)
                    assert learnable_anchor_box_3d.shape[-1] == 3
                    new_learnable_anchor_box_3d = tmp
                    new_learnable_anchor_box_3d[..., :2] = tmp[..., :2] + inverse_sigmoid(learnable_anchor_box_3d[..., :2])
                    d_ref[..., :1] = d_ref[..., :1] + inverse_sigmoid(learnable_anchor_box_3d[..., 2:3])  # d = d'+d
                    new_learnable_anchor_box_3d = torch.cat((new_learnable_anchor_box_3d.sigmoid(), d_ref[..., :1].sigmoid()), 2)
                    # new(x,y,t,b,l,r,d)

                learnable_anchor_box_3d = new_learnable_anchor_box_3d.detach()

            if self.dim_embed is not None:
                reference_dims = self.dim_embed[lid](output)  #dim_embed=MLP()

            if self.return_intermediate:
                intermediate.append(output)
                intermediate_reference_points.append(learnable_anchor_box_3d)
                intermediate_reference_dims.append(reference_dims)
                depth_reg.append(d_ref)

        if self.return_intermediate:
            return torch.stack(intermediate), torch.stack(intermediate_reference_points), torch.stack(
                intermediate_reference_dims), torch.stack(depth_reg)

        return output, reference_points


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def build_depthaware_transformer(cfg):
    return DepthAwareTransformer(
        d_model=cfg['hidden_dim'],
        dropout=cfg['dropout'],
        activation="relu",
        nhead=cfg['nheads'],
        dim_feedforward=cfg['dim_feedforward'],
        num_encoder_layers=cfg['enc_layers'],
        num_decoder_layers=cfg['dec_layers'],
        return_intermediate_dec=cfg['return_intermediate_dec'],
        num_feature_levels=cfg['num_feature_levels'],
        dec_n_points=cfg['dec_n_points'],
        enc_n_points=cfg['enc_n_points'],
        two_stage=cfg['two_stage'],
        two_stage_num_proposals=cfg['num_queries'])


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 20 ** (2 * (dim_t // 2) / 128)  # 128
    x_embed = pos_tensor[:, :, 0] * scale  # 1,300
    y_embed = pos_tensor[:, :, 1] * scale  # 1,300
    d_embed = pos_tensor[:, :, 2]
    pos_x = x_embed[:, :, None] / dim_t  # 16,50,128
    pos_y = y_embed[:, :, None] / dim_t
    pos_d = d_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)  # 沿一个新维度对输入张量序列进行连接,序列中所有张量应为相同形状
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_d = torch.stack((pos_d[:, :, 0::2].sin(), pos_d[:, :, 1::2].cos()), dim=3).flatten(2)
    # if pos_tensor.size(-1) == 2:
    pos = torch.cat((pos_y, pos_x, pos_d), dim=2)
    # elif pos_tensor.size(-1) == 6:
    #     w_embed = pos_tensor[:, :, 2] * scale
    #     pos_w = w_embed[:, :, None] / dim_t
    #     pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
    #
    #     h_embed = pos_tensor[:, :, 3] * scale
    #     pos_h = h_embed[:, :, None] / dim_t
    #     pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
    #
    #     pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    # else:
    #     raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos

def gen_sineembed_for_position_for_ref(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 20 ** (2 * (dim_t // 2) / 128)  # 128
    x_embed = pos_tensor[:, :, 0].sin() * scale  # 1,300
    y_embed = pos_tensor[:, :, 1].cos() * scale  # 1,300
    pos_x = x_embed[:, :, None]   # 16,50,128
    pos_y = y_embed[:, :, None]
    # pos_x = torch.stack((x_embed[:, :, 0:1].sin(), x_embed[:, :, 1:2].cos()), dim=3).flatten(2)  # 沿一个新维度对输入张量序列进行连接,序列中所有张量应为相同形状
    # pos_y = torch.stack((y_embed[:, :, 0:1].sin(), y_embed[:, :, 1:2].cos()), dim=3).flatten(2)  # flatten 从第dim个维度开始展开
    # if pos_tensor.size(-1) == 2:
    pos = torch.cat((pos_x, pos_y), dim=2)
    # elif pos_tensor.size(-1) == 6:
    #     w_embed = pos_tensor[:, :, 2] * scale
    #     pos_w = w_embed[:, :, None] / dim_t
    #     pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
    #
    #     h_embed = pos_tensor[:, :, 3] * scale
    #     pos_h = h_embed[:, :, None] / dim_t
    #     pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
    #
    #     pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    # else:
    #     raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos