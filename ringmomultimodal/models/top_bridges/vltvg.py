import copy
from typing import Optional, List
from mmcv.runner import BaseModule

import torch
import torch.nn.functional as F
from torch import nn, Tensor
import torch.distributed as dist
from torch.nn.parameter import Parameter

import math
from ringmomultimodal.models.builder import BRIDGE


# @BRIDGE.register_module()


def load_weights(module, prefix, weights):
    module_keys = module.state_dict().keys()
    weights_keys = [k for k in weights.keys() if prefix in k]
    update_weights = dict()
    for k in module_keys:
        prefix_k = prefix + '.' + k
        if prefix_k in weights_keys:
            update_weights[k] = weights[prefix_k]
        else:
            print(f"Weights of {k} are not pre-loaded.")
    module.load_state_dict(update_weights, strict=False)


@BRIDGE.register_module()
class VLTVGBridge(BaseModule):
    def __init__(self, num_queries=1, query_dim=256, norm_dim=256,
                 num_layers=6, num_extra_layers=1, init_cfg=None
                 ):
        super(VLTVGBridge, self).__init__(init_cfg)
        self.init_cfg = init_cfg
        weights_path = "/mnt/sdb/share1416/airalgorithm/detr-r50-referit.pth"
        weights = torch.load(weights_path, map_location='cpu')['model']
        self.input_proj = nn.Conv2d(2048, 256, kernel_size=1)
        load_weights(self.input_proj, prefix='input_proj', weights=weights)
        self.trans_encoder = VisualEncoder()
        load_weights(self.trans_encoder, prefix='transformer', weights=weights)
        self.masklinear = nn.Linear(640 * 640, 20 * 20)

        self.bert_proj = nn.Linear(768, 256)
        self.bert_output_layers = 4

        self.decoder = DecoderWithExtraEncoder(
            num_queries=num_queries,
            query_dim=query_dim,
            norm_dim=norm_dim,
            return_intermediate=True,
            num_layers=num_layers,
            num_extra_layers=num_extra_layers,
        )
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, img_feat, word_feat, img_mask, word_mask, pos_embed=None):
        img_feat = img_feat[-1]
        B, C, H, W = img_feat.shape
        zero_tensor = torch.zeros(B, H, W, device='cuda')
        img_mask = zero_tensor
        img_mask.zero_()

        img_feat, img_mask, pos_embed = self.trans_encoder(self.input_proj(img_feat), img_mask, pos_embed)
        word_feat = word_feat[-1][0]

        word_feat = self.bert_proj(word_feat)
        word_feat = word_feat.permute(1, 0, 2)
        word_mask = word_mask.to(dtype=torch.float32)
        hs = self.decoder(img_feat, img_mask, pos_embed,
                          word_feat, word_mask)
        return list(hs.transpose(1, 2))


class MultiStageDecoderLayer(BaseModule):
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.1,
                 img_feat_chunk_num=2):
        super().__init__()
        self.word_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
        self.img_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
        self.ffn = nn.Sequential(nn.Linear(d_model, dim_feedforward),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(dropout),
                                 nn.Linear(dim_feedforward, d_model))

        self.norm = _get_clones(nn.LayerNorm(d_model), 3)
        self.dropout = _get_clones(nn.Dropout(dropout), 3)

        self.img_feat_chunk_num = img_feat_chunk_num

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, vis_query, vis_query_pos, text_query_pos,
                img_feat=None, img_key_padding_mask=None, img_pos=None,
                word_feat=None, word_key_padding_mask=None, word_pos=None, layer_idx=None):

        if self.img_feat_chunk_num > 1:
            img_feat_srcs = img_feat.chunk(self.img_feat_chunk_num, dim=-1)
            img_feat_k = img_feat_srcs[1]
            img_feat_v = img_feat_srcs[0]
        else:
            img_feat_k = img_feat_v = img_feat

        text_info = self.word_attn(query=self.with_pos_embed(vis_query, vis_query_pos),
                                   key=self.with_pos_embed(word_feat, word_pos),
                                   value=word_feat, key_padding_mask=word_key_padding_mask)[0]
        text_query = self.norm[0](self.dropout[0](text_info))
        vis_info = self.img_attn(query=self.with_pos_embed(text_query, text_query_pos),
                                 key=self.with_pos_embed(img_feat_k, img_pos),
                                 value=img_feat_v, key_padding_mask=img_key_padding_mask)[0]

        vis_query = self.norm[1](vis_query + self.dropout[1](vis_info))
        vis_query = self.norm[2](vis_query + self.dropout[2](self.ffn(vis_query)))

        return vis_query


class DecoderWithExtraEncoder(BaseModule):
    def __init__(self, num_queries, query_dim,
                 num_layers, norm_dim, return_intermediate=False,
                 num_extra_layers=1):
        super().__init__()

        extra_encoder_layer = DiscriminativeFeatEncLayer(d_model=query_dim, img_query_with_pos=False)
        self.extra_encoder_layers = _get_clones(extra_encoder_layer, num_extra_layers)

        decoder_layer = MultiStageDecoderLayer(d_model=query_dim, dim_feedforward=2048, dropout=0.1,
                                               img_feat_chunk_num=2)
        self.layers = _get_clones(decoder_layer, num_layers)

        self.norm = nn.LayerNorm(norm_dim)
        self.return_intermediate = return_intermediate
        self.vis_query_embed = nn.Embedding(num_queries, query_dim)
        self.text_query_embed = nn.Embedding(num_queries, query_dim)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat, img_key_padding_mask=None, pos=None,
                word_feat=None, word_key_padding_mask=None):

        intermediate = []
        hw, bs, c = img_feat.shape

        for layer in self.extra_encoder_layers:
            img_feat = layer(img_feat, img_key_padding_mask, pos,
                             word_feat, word_key_padding_mask, None)

        vis_query_embed = self.vis_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        text_query_embed = self.text_query_embed.weight.unsqueeze(1).repeat(1, bs, 1)

        vis_query = torch.zeros_like(vis_query_embed)
        for idx, layer in enumerate(self.layers):
            vis_query = layer(vis_query, vis_query_embed, text_query_embed,
                              img_feat, img_key_padding_mask, pos,
                              word_feat, word_key_padding_mask, None, idx)
            if self.return_intermediate:
                intermediate.append(self.norm(vis_query))

        output = vis_query
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class DiscriminativeFeatEncLayer(BaseModule):
    def __init__(self, d_model, img_query_with_pos=True,
                 vl_verify=None):
        super().__init__()
        self.img2text_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)
        self.img_query_with_pos = img_query_with_pos

        self.text_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)
        self.img_proj = MLP(input_dim=256, hidden_dim=256, output_dim=256, num_layers=1)
        self.tf_pow = 2.0
        self.tf_scale = Parameter(torch.Tensor([1.0]))
        self.tf_sigma = Parameter(torch.Tensor([0.5]))

        self.img2textcond_attn = nn.MultiheadAttention(embed_dim=256, num_heads=8, dropout=0.1)

        self.img2img_attn = MHAttentionRPE(d_model=d_model, h=8, dropout=0.1, pos_x_range=[-20, 20],
                                           pos_y_range=[-20, 20], pos_index_offset=20)

        self.norm_text_cond_img = nn.LayerNorm(d_model)
        self.norm_img = nn.LayerNorm(d_model)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, img_feat, img_key_padding_mask, img_pos,
                word_feat, word_key_padding_mask, word_pos=None):
        orig_img_feat = img_feat

        img_query = img_feat + img_pos if self.img_query_with_pos else img_feat
        text_info = self.img2text_attn(
            query=img_query, key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat, key_padding_mask=word_key_padding_mask)[0]

        text_embed = self.text_proj(text_info)
        img_embed = self.img_proj(img_feat)
        verify_score = (F.normalize(img_embed, p=2, dim=-1) *
                        F.normalize(text_embed, p=2, dim=-1)).sum(dim=-1, keepdim=True)
        verify_score = self.tf_scale * \
                       torch.exp(- (1 - verify_score).pow(self.tf_pow) \
                                 / (2 * self.tf_sigma ** 2))
        text_cond_info = self.img2textcond_attn(
            query=img_feat, key=self.with_pos_embed(word_feat, word_pos),
            value=word_feat, key_padding_mask=word_key_padding_mask)[0]

        q = k = img_feat + text_cond_info
        text_cond_img_ctx = self.img2img_attn(
            query=q, key=k, value=img_feat, key_padding_mask=img_key_padding_mask)[0]
        fuse_img_feat = (self.norm_img(img_feat) +
                         self.norm_text_cond_img(text_cond_img_ctx)) * verify_score

        return torch.cat([orig_img_feat, fuse_img_feat], dim=-1)


_MODULES = {
    'DecoderWithExtraEncoder': DecoderWithExtraEncoder,
    'MultiStageDecoderLayer': MultiStageDecoderLayer,
    'DiscriminativeFeatEncLayer': DiscriminativeFeatEncLayer,
}


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu_inplace":
        return nn.ReLU(inplace=True)
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MHAttentionRPE(BaseModule):
    ''' With relative position embedding '''

    def __init__(self, d_model, h, dropout=0.1, return_raw_attention=False,
                 pos_x_range=[-20, 20], pos_y_range=[-20, 20], pos_index_offset=20,
                 learnable_pos_embed=False):
        super().__init__()
        self.d_k = d_model // h
        self.h = h
        self.scaling = float(self.d_k) ** -0.5
        self.return_raw_attention = return_raw_attention

        self.in_proj_weight = Parameter(torch.Tensor(3 * d_model, d_model))
        self.in_proj_bias = Parameter(torch.empty(3 * d_model))
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

        self.attn = None
        self.dropout_p = dropout
        self._reset_parameters()

        self.learnable_pos_embed = learnable_pos_embed
        if learnable_pos_embed:
            self.pos_x = nn.Embedding(pos_x_range[1] - pos_x_range[0] + 1, d_model // 2)
            self.pos_y = nn.Embedding(pos_y_range[1] - pos_y_range[0] + 1, d_model // 2)
        else:
            pos_x, pos_y = position_embedding_sine(d_model // 2, normalize=True,
                                                   x_range=pos_x_range, y_range=pos_y_range)
            self.register_buffer('pos_x', pos_x)  # [x_range, C]
            self.register_buffer('pos_y', pos_y)  # [y_range, C]

        self.pos_index_offset = pos_index_offset

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.in_proj_weight)
        nn.init.constant_(self.in_proj_bias, 0.)
        nn.init.constant_(self.out_proj.bias, 0.)

    def forward(self, query, key, value, key_padding_mask=None):
        tgt_len, bs, dim = query.size()
        src_len, _, dim = key.size()

        weight_q, bias_q = self.in_proj_weight[0:dim], self.in_proj_bias[0:dim]
        weight_k, bias_k = self.in_proj_weight[dim:dim * 2], self.in_proj_bias[dim:dim * 2]
        weight_v, bias_v = self.in_proj_weight[dim * 2:], self.in_proj_bias[dim * 2:]

        q = query.matmul(weight_q.t()) + bias_q
        k = key.matmul(weight_k.t()) + bias_k
        v = value.matmul(weight_v.t()) + bias_v

        q = q.view(tgt_len, bs * self.h, -1).transpose(0, 1)  # [bs*h, tgt_len, dim//h]
        k = k.view(src_len, bs * self.h, -1).permute(1, 2, 0)  # [bs*h, dim//h, src_len], To calculate qTk (bmm)
        v = v.view(src_len, bs * self.h, -1).transpose(0, 1)

        q = q * self.scaling
        attn_weights = torch.bmm(q, k)  # [bs*h, tgt_len, src_len]

        bs, HW = key_padding_mask.size()
        assert (HW == 400) and (HW == tgt_len)
        key_padding_mask = key_padding_mask.zero_().bool()

        img_mask = ~key_padding_mask.view(bs, 20, 20)
        yy = img_mask.cumsum(1, dtype=torch.float32).view(bs, -1)  # [bs, HW],  1~20
        xx = img_mask.cumsum(2, dtype=torch.float32).view(bs, -1)  # [bs, HW],  1~20
        diff_yy = yy[:, :, None] - yy[:, None, :]  # [bs, HW, HW]
        diff_xx = xx[:, :, None] - xx[:, None, :]  # [bs, HW, HW]
        if self.learnable_pos_embed:
            k_posy = self.pos_y.weight.matmul(weight_k.t()[:dim // 2])  # [x_range, dim]
            k_posx = self.pos_x.weight.matmul(weight_k.t()[dim // 2:])  # [y_range, dim]
        else:
            k_posy = self.pos_y.matmul(weight_k.t()[:dim // 2])  # [x_range, dim]
            k_posx = self.pos_x.matmul(weight_k.t()[dim // 2:])  # [y_range, dim]
        k_posy = k_posy.view(-1, 1, self.h, dim // self.h).repeat(1, bs, 1, 1). \
            reshape(-1, bs * self.h, dim // self.h).permute(1, 2, 0)  # [bs*h, dim//h, y_range]
        k_posx = k_posx.view(-1, 1, self.h, dim // self.h).repeat(1, bs, 1, 1). \
            reshape(-1, bs * self.h, dim // self.h).permute(1, 2, 0)  # [bs*h, dim//h, x_range]
        posy_attn_weights = torch.bmm(q, k_posy).view(bs, self.h, HW, -1)  # [bs, h, HW, y_range]
        posx_attn_weights = torch.bmm(q, k_posx).view(bs, self.h, HW, -1)  # [bs, h, HW, x_range]
        diff_yy_idx = diff_yy[:, None].repeat(1, self.h, 1, 1) + self.pos_index_offset
        diff_xx_idx = diff_xx[:, None].repeat(1, self.h, 1, 1) + self.pos_index_offset

        posy_attn_weights = torch.gather(posy_attn_weights, -1, diff_yy_idx.long())  # [bs, h, HW, HW]
        posx_attn_weights = torch.gather(posx_attn_weights, -1, diff_xx_idx.long())  # [bs, h, HW, HW]
        pos_attn_weights = (posy_attn_weights + posx_attn_weights).view(bs * self.h, HW, -1)
        attn_weights = attn_weights + pos_attn_weights

        if key_padding_mask is not None:
            attn_weights = attn_weights.view(-1, self.h, tgt_len, src_len)
            attn_weights = attn_weights.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2),  # [bs, 1, 1, src_len]
                float('-inf')
            )
            attn_weights = attn_weights.view(-1, tgt_len, src_len)
        raw_attn_weights = attn_weights
        attn_weights = attn_weights.softmax(dim=-1)
        attn_weights = F.dropout(attn_weights, p=self.dropout_p, training=self.training)
        attn_output = torch.bmm(attn_weights, v)
        self.attn = attn_weights

        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bs, -1)
        attn_output = F.linear(attn_output, self.out_proj.weight, self.out_proj.bias)
        if self.return_raw_attention:
            return attn_output, raw_attn_weights
        return attn_output, attn_weights


MULTIHEAD_ATTNS = {
    'MultiheadAttention': nn.MultiheadAttention,
    'MHAttentionRPE': MHAttentionRPE,
}


class MLP(BaseModule):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        if num_layers > 0:
            h = [hidden_dim] * (num_layers - 1)
            self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        else:
            self.layers = []

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def position_embedding_sine(num_pos_feats=64, temperature=10000, normalize=False, scale=None,
                            x_range=[-20, 20], y_range=[-20, 20], device=None):
    if scale is not None and normalize is False:
        raise ValueError("normalize should be True if scale is passed")
    if scale is None:
        scale = 2 * math.pi

    x_embed = torch.arange(x_range[0], x_range[1] + 1, device=device)  #
    y_embed = torch.arange(y_range[0], y_range[1] + 1, device=device)
    if normalize:
        eps = 1e-6
        y_embed = y_embed / (y_embed[-1] + eps) * scale
        x_embed = x_embed / (x_embed[-1] + eps) * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, None] / dim_t
    pos_y = y_embed[:, None] / dim_t
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=-1).flatten(1)
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=-1).flatten(1)
    return pos_x, pos_y


# transformer部分
class VisualEncoder(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=True):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        src = src.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory, mask, pos_embed


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self,
                     src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(q, k, value=src2, attn_mask=src_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src

        for layer in self.layers:
            output = layer(output, src_mask=mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos)

        if self.norm is not None:
            output = self.norm(output)

        return output
