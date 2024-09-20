from collections import OrderedDict
import torch
from typing import Optional
from torch import nn, Tensor
from ..utils.transfomer import PositionEmbeddingSine, PositionEmbeddingLearned, \
    TransformerEncoder, TransformerEncoderLayer
from mmcv.runner import BaseModule
from ringmomultimodal.models.builder import BRIDGE
from debug_tools import show_value


@BRIDGE.register_module()
class VLFBridge(BaseModule):
    def __init__(self, hidden_dim=256, dropout=0.1, nheads=8, dim_feedforward=2048, enc_layers=6, dec_layers=6,
                 pre_norm=True, position_embedding_category='learned',
                 N_steps=256, vision_channel=3584, language_channel=768, init_cfg=None
                 ):
        super(VLFBridge, self).__init__(init_cfg)
        self.init_cfg = init_cfg
        # Multimodal Fusion Module
        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            num_decoder_layers=dec_layers,
            normalize_before=pre_norm,
            # TODO: return_intermediate_dec
            return_intermediate_dec=True,
        )
        if position_embedding_category in ('v3', 'learned'):
            self.pos_embed = PositionEmbeddingLearned(N_steps)
        else:
            self.pos_embed = PositionEmbeddingSine(N_steps, normalize=True)
        self.pr = nn.Embedding(1, hidden_dim)
        self.v_proj = torch.nn.Sequential(
            nn.Linear(vision_channel, hidden_dim),
            nn.ReLU(), )
        self.l_proj = torch.nn.Sequential(
            nn.Linear(language_channel, hidden_dim),
            nn.ReLU(), )

    def init_weights(self):
        super(VLFBridge, self).init_weights()
        if self.init_cfg is not None:
            if isinstance(self.init_cfg, dict) and self.init_cfg["type"] == "Pretrained":
                dict_trained = torch.load(self.init_cfg["checkpoint"])['model']
                dict_new = OrderedDict(self.state_dict())
                for key in dict_new.keys():
                    if key in dict_trained.keys():
                        dict_new[key] = dict_trained[key]
                self.load_state_dict(dict_new, False)

    def forward(self, fv, fl, **kwargs):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        """
        if isinstance(fl, list) and len(fl) == 1:
            fl = fl[0]
        feature_language, pooled_output = fl
        fl = feature_language
        if isinstance(fl, list):
            fl = sum(fl) / len(fl)
        if isinstance(fv, list):
            assert len(fv) > 0
            fv = fv[-1]

        bs, c, h, w = fv.shape
        _, _, l = fl.shape

        pv = self.v_proj(fv.view(bs, c, -1).permute(0, 2, 1))
        pl = self.l_proj(fl)
        pv = pv.permute(0, 2, 1)
        pl = pl.permute(0, 2, 1)

        pr = self.pr.weight
        pr = pr.expand(bs, -1).unsqueeze(2)

        x0 = torch.cat((pv, pl), dim=2)
        x0 = torch.cat((x0, pr), dim=2)

        pos = self.pos_embed(x0).to(x0.dtype)
        mask = torch.zeros([bs, x0.shape[2]]).cuda()
        mask = mask.bool()
        out = self.transformer(x0, mask, pos)

        return [out[-1]]


class Transformer(BaseModule):
    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
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
        # flatten NxCxHxW to HWxNxC
        # permute NxCxW to WxNxC
        src = src.permute(2, 0, 1)
        pos_embed = pos_embed.permute(1, 0, 2)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory
