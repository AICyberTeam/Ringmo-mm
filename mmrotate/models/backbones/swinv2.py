# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import OrderedDict
from copy import deepcopy
import random
import torch.backends.cudnn as cudnn
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from mmcv.cnn import build_norm_layer
from mmcv.cnn.bricks.transformer import FFN, build_dropout

from mmcv.runner import BaseModule, ModuleList, _load_checkpoint
from mmcv.cnn import build_norm_layer, constant_init, trunc_normal_init
from mmcv.utils import to_2tuple
from ..builder import ROTATED_BACKBONES as BACKBONES
from ...utils import get_root_logger




# from ..layers import PatchEmbed, PatchMerging
from ..utils import PatchEmbed_V2, PatchMerging_V2
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class WindowMSA(BaseModule):
    """Window based multi-head self-attention (W-MSA) module with relative
    position bias.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (tuple[int]): The height and width of the window.
        qkv_bias (bool, optional):  If True, add a learnable bias to q, k, v.
            Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Default: 0.0
        proj_drop_rate (float, optional): Dropout ratio of output. Default: 0.
        init_cfg (dict | None, optional): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 num_heads,
                 window_size,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0.,
                 proj_drop_rate=0.,
                 init_cfg=None,
                 pretrained_window_size=[0,0]):

        super().__init__()
        self.embed_dims = embed_dims
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_embed_dims = embed_dims // num_heads
        self.scale = qk_scale or head_embed_dims**-0.5
        self.init_cfg = init_cfg
        self.pretrained_window_size = pretrained_window_size
        # # define a parameter table of relative position bias
        # self.relative_position_bias_table = nn.Parameter(
        #     torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1),
        #                 num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # # About 2x faster than original impl
        # Wh, Ww = self.window_size
        # rel_index_coords = self.double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        # rel_position_index = rel_index_coords + rel_index_coords.T
        # rel_position_index = rel_position_index.flip(1).contiguous()
        # self.register_buffer('relative_position_index', rel_position_index)


        # mlp to generate continuous relative position bias
        self.cpb_mlp = nn.Sequential(nn.Linear(2, 512, bias=True),
                                     nn.ReLU(inplace=True),
                                     nn.Linear(512, num_heads, bias=False))

        # get relative_coords_table
        relative_coords_h = torch.arange(-(self.window_size[0] - 1), self.window_size[0], dtype=torch.float32)
        relative_coords_w = torch.arange(-(self.window_size[1] - 1), self.window_size[1], dtype=torch.float32)
        relative_coords_table = torch.stack(
            torch.meshgrid([relative_coords_h,
                            relative_coords_w])).permute(1, 2, 0).contiguous().unsqueeze(0)  # 1, 2*Wh-1, 2*Ww-1, 2
        if pretrained_window_size[0] > 0:
            relative_coords_table[:, :, :, 0] /= (pretrained_window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (pretrained_window_size[1] - 1)
        else:
            relative_coords_table[:, :, :, 0] /= (self.window_size[0] - 1)
            relative_coords_table[:, :, :, 1] /= (self.window_size[1] - 1)
        relative_coords_table *= 8  # normalize to -8, 8
        relative_coords_table = torch.sign(relative_coords_table) * torch.log2(
            torch.abs(relative_coords_table) + 1.0) / np.log2(8)

        self.register_buffer("relative_coords_table", relative_coords_table)

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(embed_dims, embed_dims * 3, bias=False)
        if qkv_bias:
            self.q_bias = nn.Parameter(torch.zeros(embed_dims))
            self.v_bias = nn.Parameter(torch.zeros(embed_dims))
        else:
            self.q_bias = None
            self.v_bias = None
        self.attn_drop = nn.Dropout(attn_drop_rate)
        self.proj = nn.Linear(embed_dims, embed_dims)
        self.proj_drop = nn.Dropout(proj_drop_rate)

        self.softmax = nn.Softmax(dim=-1)
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))), requires_grad=True)

    # def init_weights(self):
    #     trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x, mask=None):
        """
        Args:

            x (tensor): input features with shape of (num_windows*B, N, C)
            mask (tensor | None, Optional): mask with shape of (num_windows,
                Wh*Ww, Wh*Ww), value should be between (-inf, 0].
        """
        B, N, C = x.shape
        if self.q_bias is not None:
            qkv_bias = torch.cat((self.q_bias, torch.zeros_like(self.v_bias, requires_grad=False), self.v_bias))
        qkv = F.linear(input=x, weight=self.qkv.weight, bias=qkv_bias)
        qkv = qkv.reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # cosine attention
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))
        threshold = torch.log(torch.tensor(1. / 0.01)).to(self.logit_scale.device)
        logit_scale = torch.clamp(self.logit_scale, max=threshold).exp()
        attn = attn * logit_scale

        relative_position_bias_table = self.cpb_mlp(self.relative_coords_table).view(-1, self.num_heads)
        relative_position_bias = relative_position_bias_table[
            self.relative_position_index.view(-1)].view(
                self.window_size[0] * self.window_size[1],
                self.window_size[0] * self.window_size[1],
                -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    @staticmethod
    def double_step_seq(step1, len1, step2, len2):
        seq1 = torch.arange(0, step1 * len1, step1)
        seq2 = torch.arange(0, step2 * len2, step2)
        return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


class ShiftWindowMSA(BaseModule):
    """Shifted Window Multihead Self-Attention Module.

    Args:
        embed_dims (int): Number of input channels.
        num_heads (int): Number of attention heads.
        window_size (int): The height and width of the window.
        shift_size (int, optional): The shift step of each window towards
            right-bottom. If zero, act as regular window-msa. Defaults to 0.
        qkv_bias (bool, optional): If True, add a learnable bias to q, k, v.
            Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Defaults: None.
        attn_drop_rate (float, optional): Dropout ratio of attention weight.
            Defaults: 0.
        proj_drop_rate (float, optional): Dropout ratio of output.
            Defaults: 0.
        dropout_layer (dict, optional): The dropout_layer used before output.
            Defaults: dict(type='DropPath', drop_prob=0.).
        init_cfg (dict, optional): The extra config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size=0,
                 qkv_bias=True,
                 qk_scale=None,
                 attn_drop_rate=0,
                 proj_drop_rate=0,
                 dropout_layer=dict(type='DropPath', drop_prob=0.),
                 init_cfg=None,
                 pretrained_window_size=0):
        super().__init__(init_cfg)
        self.input_resolution = input_resolution

        self.window_size = window_size
        self.shift_size = shift_size
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size

        self.w_msa = WindowMSA(
            embed_dims=embed_dims,
            num_heads=num_heads,
            window_size=to_2tuple(self.window_size),
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=proj_drop_rate,
            init_cfg=None,
        pretrained_window_size=to_2tuple(pretrained_window_size))

        self.drop = build_dropout(dropout_layer)

    def forward(self, query, hw_shape):
        B, L, C = query.shape
        H, W = hw_shape
        assert L == H * W, 'input feature has wrong size'
        query = query.view(B, H, W, C)

        # pad feature maps to multiples of window size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        query = F.pad(query, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = query.shape[1], query.shape[2]

        # cyclic shift
        if self.shift_size > 0:
            shifted_query = torch.roll(
                query,
                shifts=(-self.shift_size, -self.shift_size),
                dims=(1, 2))

            # calculate attention mask for SW-MSA
            img_mask = torch.zeros((1, H_pad, W_pad, 1), device=query.device)
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size,
                              -self.shift_size), slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = self.window_partition(img_mask)
            mask_windows = mask_windows.view(
                -1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0,
                                              float(-100.0)).masked_fill(
                                                  attn_mask == 0, float(0.0))
        else:
            shifted_query = query
            attn_mask = None

        # nW*B, window_size, window_size, C
        query_windows = self.window_partition(shifted_query)
        # nW*B, window_size*window_size, C
        query_windows = query_windows.view(-1, self.window_size**2, C)

        # W-MSA/SW-MSA (nW*B, window_size*window_size, C)
        attn_windows = self.w_msa(query_windows, mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size,
                                         self.window_size, C)

        # B H' W' C
        shifted_x = self.window_reverse(attn_windows, H_pad, W_pad)
        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2))
        else:
            x = shifted_x

        if pad_r > 0 or pad_b:
            x = x[:, :H, :W, :].contiguous()

        x = x.view(B, H * W, C)

        # x = self.drop(x)
        return x

    def window_reverse(self, windows, H, W):
        """
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            H (int): Height of image
            W (int): Width of image
        Returns:
            x: (B, H, W, C)
        """
        window_size = self.window_size
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H // window_size, W // window_size, window_size,
                         window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def window_partition(self, x):
        """
        Args:
            x: (B, H, W, C)
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        B, H, W, C = x.shape
        window_size = self.window_size
        x = x.view(B, H // window_size, window_size, W // window_size,
                   window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        windows = windows.view(-1, window_size, window_size, C)
        return windows


class SwinBlock(BaseModule):
    """"
    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        window_size (int, optional): The local window scale. Default: 7.
        shift (bool, optional): whether to shift window or not. Default False.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float, optional): Stochastic depth rate. Default: 0.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 input_resolution,
                 num_heads,
                 feedforward_channels,
                 window_size=7,
                 shift=False,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 pretrained_window_size=0):

        super(SwinBlock, self).__init__()
        self.input_resolution = input_resolution
        self.init_cfg = init_cfg
        self.with_cp = with_cp
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()
        self.norm1 = build_norm_layer(norm_cfg, embed_dims)[1]
        self.attn = ShiftWindowMSA(
            embed_dims=embed_dims,
            input_resolution = input_resolution,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=window_size // 2 if shift else 0,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop_rate=attn_drop_rate,
            proj_drop_rate=drop_rate,
            dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
            init_cfg=None,
        pretrained_window_size=pretrained_window_size)

        self.norm2 = build_norm_layer(norm_cfg, embed_dims)[1]
        # self.ffn = FFN(
        #     embed_dims=embed_dims,
        #     feedforward_channels=feedforward_channels,
        #     num_fcs=2,
        #     ffn_drop=drop_rate,
        #     dropout_layer=dict(type='DropPath', drop_prob=drop_path_rate),
        #     act_cfg=act_cfg,
        #     add_identity=True,
        #     init_cfg=None)
        self.mlp = Mlp(in_features=embed_dims, hidden_features=feedforward_channels, act_layer=nn.GELU, drop=drop_rate)
    def forward(self, x, hw_shape):

        def _inner_forward(x):
            identity = x
            x = self.attn(x, hw_shape)
            # print('4'*30, x)
            x = self.norm1(x)
            x = self.drop_path(x)
            x = x + identity
            # print('5'*30, x)

            ### FFN 
            x = x + self.drop_path(self.norm2(self.mlp(x)))
            # print('6'*30, x)

            return x

        if self.with_cp and x.requires_grad:
            x = cp.checkpoint(_inner_forward, x)
        else:
            x = _inner_forward(x)

        return x


class SwinBlockSequence(BaseModule):
    """Implements one stage in Swin Transformer.

    Args:
        embed_dims (int): The feature dimension.
        num_heads (int): Parallel attention heads.
        feedforward_channels (int): The hidden dimension for FFNs.
        depth (int): The number of blocks in this stage.
        window_size (int, optional): The local window scale. Default: 7.
        qkv_bias (bool, optional): enable bias for qkv if True. Default: True.
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        drop_rate (float, optional): Dropout rate. Default: 0.
        attn_drop_rate (float, optional): Attention dropout rate. Default: 0.
        drop_path_rate (float | list[float], optional): Stochastic depth
            rate. Default: 0.
        downsample (BaseModule | None, optional): The downsample operation
            module. Default: None.
        act_cfg (dict, optional): The config dict of activation function.
            Default: dict(type='GELU').
        norm_cfg (dict, optional): The config dict of normalization.
            Default: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        init_cfg (dict | list | None, optional): The init config.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 input_resolution,
                 num_heads,
                 feedforward_channels,
                 depth,
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 downsample=None,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 init_cfg=None,
                 pretrained_window_size=0):
        super().__init__(init_cfg=init_cfg)

        if isinstance(drop_path_rate, list):
            drop_path_rates = drop_path_rate
            assert len(drop_path_rates) == depth
        else:
            drop_path_rates = [deepcopy(drop_path_rate) for _ in range(depth)]
        self.input_resolution = input_resolution
        self.blocks = ModuleList()
        for i in range(depth):
            block = SwinBlock(
                embed_dims=embed_dims,
                input_resolution =input_resolution,
                num_heads=num_heads,
                feedforward_channels=feedforward_channels,
                window_size=window_size,
                shift=False if i % 2 == 0 else True,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rates[i],
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
            pretrained_window_size=pretrained_window_size)
            self.blocks.append(block)

        self.downsample = downsample

    def forward(self, x, hw_shape):
        for block in self.blocks:
            x = block(x, hw_shape)

        if self.downsample:
            x_down, down_hw_shape = self.downsample(x, hw_shape)
            return x_down, down_hw_shape, x, hw_shape
        else:
            return x, hw_shape, x, hw_shape


@BACKBONES.register_module()
class SwinTransformerV2(BaseModule):
    """ Swin Transformer
    A PyTorch implement of : `Swin Transformer:
    Hierarchical Vision Transformer using Shifted Windows`  -
        https://arxiv.org/abs/2103.14030

    Inspiration from
    https://github.com/microsoft/Swin-Transformer

    Args:
        pretrain_img_size (int | tuple[int]): The size of input image when
            pretrain. Defaults: 224.
        in_channels (int): The num of input channels.
            Defaults: 3.
        embed_dims (int): The feature dimension. Default: 96.
        patch_size (int | tuple[int]): Patch size. Default: 4.
        window_size (int): Window size. Default: 7.
        mlp_ratio (int): Ratio of mlp hidden dim to embedding dim.
            Default: 4.
        depths (tuple[int]): Depths of each Swin Transformer stage.
            Default: (2, 2, 6, 2).
        num_heads (tuple[int]): Parallel attention heads of each Swin
            Transformer stage. Default: (3, 6, 12, 24).
        strides (tuple[int]): The patch merging or patch embedding stride of
            each Swin Transformer stage. (In swin, we set kernel size equal to
            stride.) Default: (4, 2, 2, 2).
        out_indices (tuple[int]): Output from which stages.
            Default: (0, 1, 2, 3).
        qkv_bias (bool, optional): If True, add a learnable bias to query, key,
            value. Default: True
        qk_scale (float | None, optional): Override default qk scale of
            head_dim ** -0.5 if set. Default: None.
        patch_norm (bool): If add a norm layer for patch embed and patch
            merging. Default: True.
        drop_rate (float): Dropout rate. Defaults: 0.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Defaults: 0.1.
        use_abs_pos_embed (bool): If True, add absolute position embedding to
            the patch embedding. Defaults: False.
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='GELU').
        norm_cfg (dict): Config dict for normalization layer at
            output of backone. Defaults: dict(type='LN').
        with_cp (bool, optional): Use checkpoint or not. Using checkpoint
            will save some memory while slowing down the training speed.
            Default: False.
        pretrained (str, optional): model pretrained path. Default: None.
        convert_weights (bool): The flag indicates whether the
            pre-trained model is from the original repo. We may need
            to convert some keys to make it compatible.
            Default: False.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            Default: -1 (-1 means not freezing any parameters).
        init_cfg (dict, optional): The Config for initialization.
            Defaults to None.
    """

    def __init__(self,
                 pretrain_img_size=192,
                 in_channels=3,
                 embed_dims=96,
                 patch_size=4,
                 window_size=7,
                 mlp_ratio=4,
                 depths=(2, 2, 6, 2),
                 num_heads=(3, 6, 12, 24),
                 strides=(4, 2, 2, 2),
                 out_indices=(0, 1, 2, 3),
                 qkv_bias=True,
                 qk_scale=None,
                 patch_norm=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 use_abs_pos_embed=False,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN'),
                 with_cp=False,
                 pretrained=None,
                 convert_weights=False,
                 frozen_stages=-1,
                 init_cfg=None,
                 pretrained_window_sizes=[0,0,0,0]):
        self.convert_weights = convert_weights
        self.frozen_stages = frozen_stages
        if isinstance(pretrain_img_size, int):
            pretrain_img_size = to_2tuple(pretrain_img_size)
        elif isinstance(pretrain_img_size, tuple):
            if len(pretrain_img_size) == 1:
                pretrain_img_size = to_2tuple(pretrain_img_size[0])
            assert len(pretrain_img_size) == 2, \
                f'The size of image should have length 1 or 2, ' \
                f'but got {len(pretrain_img_size)}'

        assert not (init_cfg and pretrained), \
            'init_cfg and pretrained cannot be specified at the same time'
        if isinstance(pretrained, str):
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            self.init_cfg = dict(type='Pretrained', checkpoint=pretrained)
        elif pretrained is None:
            self.init_cfg = init_cfg
        else:
            raise TypeError('pretrained must be a str or None')

        super(SwinTransformerV2, self).__init__(init_cfg=init_cfg)

        num_layers = len(depths)
        self.out_indices = out_indices
        self.use_abs_pos_embed = use_abs_pos_embed

        assert strides[0] == patch_size, 'Use non-overlapping patch embed.'

        self.patch_embed = PatchEmbed_V2(
            in_channels=in_channels,
            embed_dims=embed_dims,
            conv_type='Conv2d',
            kernel_size=patch_size,
            stride=strides[0],
            norm_cfg=norm_cfg if patch_norm else None,
            init_cfg=None)
        patches_resolution = [pretrain_img_size[0] // patch_size, pretrain_img_size[1] // patch_size]
        if self.use_abs_pos_embed:
            patch_row = pretrain_img_size[0] // patch_size
            patch_col = pretrain_img_size[1] // patch_size
            num_patches = patch_row * patch_col
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros((1, num_patches, embed_dims)))

        self.drop_after_pos = nn.Dropout(p=drop_rate)

        # set stochastic depth decay rule
        total_depth = sum(depths)
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, total_depth)
        ]

        self.stages = ModuleList()
        in_channels = embed_dims
        for i in range(num_layers):
            if i < num_layers - 1:
                downsample = PatchMerging_V2(
                    in_channels=in_channels,
                    out_channels=2 * in_channels,
                    stride=strides[i + 1],
                    norm_cfg=norm_cfg if patch_norm else None,
                    init_cfg=None)
            else:
                downsample = None

            stage = SwinBlockSequence(
                embed_dims=in_channels,
                input_resolution=(patches_resolution[0] // (2 ** i),
                                  patches_resolution[1] // (2 ** i)),
                num_heads=num_heads[i],
                feedforward_channels=mlp_ratio * in_channels,
                depth=depths[i],
                window_size=window_size,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                downsample=downsample,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                with_cp=with_cp,
                init_cfg=None,
            pretrained_window_size=pretrained_window_sizes[i])
            self.stages.append(stage)
            if downsample:
                in_channels = downsample.out_channels

        self.num_features = [int(embed_dims * 2**(i)) for i in range(num_layers)]
        # Add a norm layer for each output
        for i in out_indices:
            layer = build_norm_layer(norm_cfg, self.num_features[i])[1]
            layer_name = f'norm{i}'
            self.add_module(layer_name, layer)

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(SwinTransformerV2, self).train(mode)
        self._freeze_stages()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            if self.use_abs_pos_embed:
                self.absolute_pos_embed.requires_grad = False
            self.drop_after_pos.eval()

        for i in range(1, self.frozen_stages + 1):

            if (i - 1) in self.out_indices:
                norm_layer = getattr(self, f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.stages[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def init_weights(self, pretrained=True):
        logger = get_root_logger()
        if self.init_cfg is None:
            logger.warn(f'No pre-trained weights for '
                        f'{self.__class__.__name__}, '
                        f'training start from scratch')
            if self.use_abs_pos_embed:
                trunc_normal_(self.absolute_pos_embed, std=0.02)
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)
                elif isinstance(m, nn.LayerNorm):
                    constant_init(m, 1.0)
        else:
            assert 'checkpoint' in self.init_cfg, f'Only support ' \
                                                  f'specify `Pretrained` in ' \
                                                  f'`init_cfg` in ' \
                                                  f'{self.__class__.__name__} '
            ckpt = _load_checkpoint(
                self.init_cfg.checkpoint, logger=logger, map_location='cpu')
            if 'state_dict' in ckpt:
                _state_dict = ckpt['state_dict']
            elif 'model' in ckpt:
                _state_dict = ckpt['model']
            else:
                _state_dict = ckpt
            if self.convert_weights:
                # supported loading weight from original repo,
                _state_dict = swin_converter(_state_dict)

            state_dict = OrderedDict()
            for k, v in _state_dict.items():
                if k.startswith('backbone.'):
                    state_dict[k[9:]] = v

            # strip prefix of state_dict
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}

            # reshape absolute position embedding
            if state_dict.get('absolute_pos_embed') is not None:
                absolute_pos_embed = state_dict['absolute_pos_embed']
                N1, L, C1 = absolute_pos_embed.size()
                N2, C2, H, W = self.absolute_pos_embed.size()
                if N1 != N2 or C1 != C2 or L != H * W:
                    logger.warning('Error in loading absolute_pos_embed, pass')
                else:
                    state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
                        N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
            all_keys = list(state_dict.keys())
            for key in all_keys:
                if "relative_position_bias_table" in key:
                    relative_position_bias_table_pretrained = state_dict[key]
                    relative_position_bias_table_current = self.state_dict[key]
                    L1, nH1 = relative_position_bias_table_pretrained.size()
                    L2, nH2 = relative_position_bias_table_current.size()
                    if nH1 != nH2:
                        logger.info(f"Error in loading {key}, passing......")
                    else:
                        if L1 != L2:
                            logger.info(f"{key}: Interpolate relative_position_bias_table using geo.")
                            src_size = int(L1 ** 0.5)
                            dst_size = int(L2 ** 0.5)

                            def geometric_progression(a, r, n):
                                return a * (1.0 - r ** n) / (1.0 - r)

                            left, right = 1.01, 1.5
                            while right - left > 1e-6:
                                q = (left + right) / 2.0
                                gp = geometric_progression(1, q, src_size // 2)
                                if gp > dst_size // 2:
                                    right = q
                                else:
                                    left = q

                            # if q > 1.090307:
                            #     q = 1.090307

                            dis = []
                            cur = 1
                            for i in range(src_size // 2):
                                dis.append(cur)
                                cur += q ** (i + 1)

                            r_ids = [-_ for _ in reversed(dis)]

                            x = r_ids + [0] + dis
                            y = r_ids + [0] + dis

                            t = dst_size // 2.0
                            dx = np.arange(-t, t + 0.1, 1.0)
                            dy = np.arange(-t, t + 0.1, 1.0)

                            logger.info("Original positions = %s" % str(x))
                            logger.info("Target positions = %s" % str(dx))

                            all_rel_pos_bias = []

                            for i in range(nH1):
                                z = relative_position_bias_table_pretrained[:, i].view(src_size,
                                                                                       src_size).float().numpy()
                                f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
                                all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
                                    relative_position_bias_table_pretrained.device))

                            new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
                            state_dict[key] = new_rel_pos_bias

            # delete relative_position_index since we always re-init it
            # relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
            # for k in relative_position_index_keys:
            #     del state_dict[k]
            #
            # # delete relative_coords_table since we always re-init it
            # relative_coords_table_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
            # for k in relative_coords_table_keys:
            #     del state_dict[k]
            #
            # # delete attn_mask since we always re-init it
            # attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
            # for k in attn_mask_keys:
            #     del state_dict[k]
            # interpolate position bias table if needed
            # relative_position_bias_table_keys = [
            #     k for k in state_dict.keys()
            #     if 'relative_position_bias_table' in k
            # ]
            # for table_key in relative_position_bias_table_keys:
            #     table_pretrained = state_dict[table_key]
            #     table_current = self.state_dict()[table_key]
            #     L1, nH1 = table_pretrained.size()
            #     L2, nH2 = table_current.size()
            #     if nH1 != nH2:
            #         logger.warning(f'Error in loading {table_key}, pass')
            #     elif L1 != L2:
            #         S1 = int(L1**0.5)
            #         S2 = int(L2**0.5)
            #         table_pretrained_resized = F.interpolate(
            #             table_pretrained.permute(1, 0).reshape(1, nH1, S1, S1),
            #             size=(S2, S2),
            #             mode='bicubic')
            #         state_dict[table_key] = table_pretrained_resized.view(
            #             nH2, L2).permute(1, 0).contiguous()

            # load state_dict
            self.load_state_dict(state_dict, False)

    def forward(self, x):
        x, hw_shape = self.patch_embed(x)
        # print(x)
        if self.use_abs_pos_embed:
            x = x + self.absolute_pos_embed
        x = self.drop_after_pos(x)
        outs = []
        for i, stage in enumerate(self.stages):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            # print(x)
            # print(x.shape)
            if i in self.out_indices:
                norm_layer = getattr(self, f'norm{i}')
                out = norm_layer(out)
                out = out.view(-1, *out_hw_shape,
                               self.num_features[i]).permute(0, 3, 1,
                                                             2).contiguous()
                outs.append(out)

        return outs


def swin_converter(ckpt):

    new_ckpt = OrderedDict()

    def correct_unfold_reduction_order(x):
        out_channel, in_channel = x.shape
        x = x.reshape(out_channel, 4, in_channel // 4)
        x = x[:, [0, 2, 1, 3], :].transpose(1,2).reshape(out_channel, in_channel)
        return x
    #
    # def correct_unfold_norm_order(x):
    #     in_channel = x.shape[0]
    #     x = x.reshape(2, in_channel // 2)
    #     x = x[[1,0], :].transpose(0, 1).reshape(in_channel)
    #     return x

    # def correct_unfold_norm_order(x):
    #     in_channel = x.shape[0]
    #     x = x.reshape(4, in_channel // 4)
    #     x = x[[0, 2, 1, 3], :].transpose(0, 1).reshape(in_channel)
    #     return x

    new_state_dict1 = OrderedDict()
    for k, v in ckpt.items():
        if k.startswith('encoder.'):
            k = k[8:]
        elif k.startswith('backbone.'):
            k = k[9:]
        elif k.startswith('module.'):
            k = k[7:]
        new_state_dict1[k] = v

    for k, v in new_state_dict1.items():
        if k.startswith('head'):
            continue
        elif k.startswith('layers'):
            new_v = v
            if 'attn.' in k:
                new_k = k.replace('attn.', 'attn.w_msa.')
                if 'rpe_mlp.' in new_k:
                    new_k = new_k.replace('rpe_mlp.', 'cpb_mlp.')
            # elif 'mlp.' in k:
            #     if 'mlp.fc1.' in k:
            #         new_k = k.replace('mlp.fc1.', 'ffn.layers.0.0.')
            #     elif 'mlp.fc2.' in k:
            #         new_k = k.replace('mlp.fc2.', 'ffn.layers.1.')
            #     else:
            #         new_k = k.replace('mlp.', 'ffn.')
            elif 'downsample' in k:
                new_k = k
                if 'reduction.' in k:
                    new_v = correct_unfold_reduction_order(v)
                # elif 'norm.' in k:
                #     new_v = correct_unfold_norm_order(v)
            else:
                new_k = k
            new_k = new_k.replace('layers', 'stages', 1)
        elif k.startswith('patch_embed'):
            new_v = v
            if 'proj' in k:
                new_k = k.replace('proj', 'projection')
            else:
                new_k = k
        # elif k.startswith('norm.'):
        #     new_k = k.replace('norm.','norm3.')
        #     new_v = v
        
        #elif k.startswith('norm.'):
        #    new_k = k
        #    new_k0 = k.replace('norm.','norm0.')
        #    # new_k1 = k.replace('norm.','norm1.')
        #    # new_k2 = k.replace('norm.','norm2.')
        #    new_v = v
#
        #    new_ckpt['backbone.' + new_k0] = new_v
            # new_ckpt['backbone.' + new_k1] = new_v
            # new_ckpt['backbone.' + new_k2] = new_v
        else:
            new_v = v
            new_k = k

        new_ckpt['backbone.' + new_k] = new_v

    return new_ckpt

# if __name__ == '__main__':
#     random.seed(321)
#     np.random.seed(321)
#     torch.manual_seed(321)
#     torch.cuda.manual_seed(321)
#     torch.cuda.manual_seed_all(321)
#     cudnn.benchmark = False
#     cudnn.deterministic = True

#     Swinv2 = RingMoGiant(pretrain_img_size=192, patch_size=4, in_channels=3,
#                  embed_dims=448, depths=[2, 2, 18, 2], num_heads=[14, 28, 56, 112],
#                  window_size=12).cuda()
#     checkpoint = torch.load('/mnt/zhisuan_nfs/FoundationModel/Swin-Transformer-main/PretrainedModel/swinv2_giant_22k_500k.pth', map_location='cpu')
#     _state_dict=swin_converter(checkpoint['model'])
#     state_dict = OrderedDict()
#     for k, v in _state_dict.items():
#         if k.startswith('backbone.'):
#             state_dict[k[9:]] = v

#     # strip prefix of state_dict
#     if list(state_dict.keys())[0].startswith('module.'):
#         state_dict = {k[7:]: v for k, v in state_dict.items()}

#     # # reshape absolute position embedding
#     # if state_dict.get('absolute_pos_embed') is not None:
#     #     absolute_pos_embed = state_dict['absolute_pos_embed']
#     #     N1, L, C1 = absolute_pos_embed.size()
#     #     N2, C2, H, W = self.absolute_pos_embed.size()
#     #     if N1 != N2 or C1 != C2 or L != H * W:
#     #         logger.warning('Error in loading absolute_pos_embed, pass')
#     #     else:
#     #         state_dict['absolute_pos_embed'] = absolute_pos_embed.view(
#     #             N2, H, W, C2).permute(0, 3, 1, 2).contiguous()
#     # all_keys = list(state_dict.keys())
#     # for key in all_keys:
#     #     if "relative_position_bias_table" in key:
#     #         relative_position_bias_table_pretrained = state_dict[key]
#     #         relative_position_bias_table_current = self.state_dict[key]
#     #         L1, nH1 = relative_position_bias_table_pretrained.size()
#     #         L2, nH2 = relative_position_bias_table_current.size()
#     #         if nH1 != nH2:
#     #             logger.info(f"Error in loading {key}, passing......")
#     #         else:
#     #             if L1 != L2:
#     #                 logger.info(f"{key}: Interpolate relative_position_bias_table using geo.")
#     #                 src_size = int(L1 ** 0.5)
#     #                 dst_size = int(L2 ** 0.5)
#     #
#     #
#     #                 def geometric_progression(a, r, n):
#     #                     return a * (1.0 - r ** n) / (1.0 - r)
#     #
#     #
#     #                 left, right = 1.01, 1.5
#     #                 while right - left > 1e-6:
#     #                     q = (left + right) / 2.0
#     #                     gp = geometric_progression(1, q, src_size // 2)
#     #                     if gp > dst_size // 2:
#     #                         right = q
#     #                     else:
#     #                         left = q
#     #
#     #                 # if q > 1.090307:
#     #                 #     q = 1.090307
#     #
#     #                 dis = []
#     #                 cur = 1
#     #                 for i in range(src_size // 2):
#     #                     dis.append(cur)
#     #                     cur += q ** (i + 1)
#     #
#     #                 r_ids = [-_ for _ in reversed(dis)]
#     #
#     #                 x = r_ids + [0] + dis
#     #                 y = r_ids + [0] + dis
#     #
#     #                 t = dst_size // 2.0
#     #                 dx = np.arange(-t, t + 0.1, 1.0)
#     #                 dy = np.arange(-t, t + 0.1, 1.0)
#     #
#     #                 logger.info("Original positions = %s" % str(x))
#     #                 logger.info("Target positions = %s" % str(dx))
#     #
#     #                 all_rel_pos_bias = []
#     #
#     #                 for i in range(nH1):
#     #                     z = relative_position_bias_table_pretrained[:, i].view(src_size,
#     #                                                                            src_size).float().numpy()
#     #                     f_cubic = interpolate.interp2d(x, y, z, kind='cubic')
#     #                     all_rel_pos_bias.append(torch.Tensor(f_cubic(dx, dy)).contiguous().view(-1, 1).to(
#     #                         relative_position_bias_table_pretrained.device))
#     #
#     #                 new_rel_pos_bias = torch.cat(all_rel_pos_bias, dim=-1)
#     #                 state_dict[key] = new_rel_pos_bias

#     # delete relative_position_index since we always re-init it
#     # relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
#     # for k in relative_position_index_keys:
#     #     del state_dict[k]
#     #
#     # # delete relative_coords_table since we always re-init it
#     # relative_coords_table_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
#     # for k in relative_coords_table_keys:
#     #     del state_dict[k]
#     #
#     # # delete attn_mask since we always re-init it
#     # attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
#     # for k in attn_mask_keys:
#     #     del state_dict[k]


#     Swinv2.load_state_dict(state_dict, strict=True)
#     x= torch.ones(1,3,192,192).cuda()
#     outout=Swinv2(x)
#     # print(outout)