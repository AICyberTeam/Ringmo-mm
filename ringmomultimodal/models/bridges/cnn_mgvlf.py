from ringmomultimodal.models.bridges.base import RingMoMultiModalBridgeBase
import torch.nn as nn
from ringmomultimodal.models.utils.transfomer import TransformerEncoderLayer, TransformerEncoder, TransformerDecoder, \
    PositionEmbeddingLearned, PositionEmbeddingSine, TransformerDecoderLayer
from ringmomultimodal.models.utils.misc import NestedTensor
import torch
from collections import OrderedDict
from typing import Dict, List
import torch.nn.functional as F
from ..builder import BRIDGE


@BRIDGE.register_module()
class CNNMGVLFBridge(RingMoMultiModalBridgeBase):
    def __init__(self,
                 hidden_dim=256,
                 dropout=0.1,
                 nheads=8,
                 dim_feedforward=2048,
                 enc_layers=6,
                 pre_norm=True,
                 position_embedding_category='sine',
                 in_channel=2048,
                 init_cfg=None):

        super(CNNMGVLFBridge, self).__init__(init_cfg)

        self.pos_in = generate_position_embedding(hidden_dim, position_embedding_category)
        self.pos = generate_position_embedding(hidden_dim, position_embedding_category)
        self.text_pos_embed = nn.Embedding(40 + 1, hidden_dim)

        self.transformer = Transformer(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_encoder_layers=enc_layers,
            normalize_before=pre_norm,
            # TODO: return_intermediate_dec
        )

        self.DE = Transformer_Decoder(
            d_model=hidden_dim,
            dropout=dropout,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            num_decoder_layers=1,
            normalize_before=pre_norm,
            return_intermediate_dec=True
        )

        self.conv6_1 = nn.Conv2d(in_channels=in_channel, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.conv6_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2))
        self.conv7_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.conv7_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2))
        self.conv8_1 = nn.Conv2d(in_channels=256, out_channels=128, kernel_size=(1, 1), stride=(1, 1))
        self.conv8_2 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 3), stride=(2, 2))
        self.adp_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.input_proj = nn.Conv2d(in_channel, hidden_dim, kernel_size=(1, 1))
        self.l_proj = torch.nn.Sequential(nn.Linear(768, hidden_dim), nn.ReLU(), )

    def init_weights(self):
        super(CNNMGVLFBridge, self).init_weights()
        if self.init_cfg is not None:
            if isinstance(self.init_cfg, dict) and self.init_cfg["type"] == "Pretrained":
                dict_trained = torch.load(self.init_cfg["checkpoint"])['model']
                dict_new = OrderedDict(self.state_dict())
                for key in dict_new.keys():
                    if key in dict_trained.keys():
                        dict_new[key] = dict_trained[key]
                self.load_state_dict(dict_new, False)

    def _masks_generate(self, xs, img_mask):
        out: List[NestedTensor] = []
        for x in xs:
            if isinstance(x, tuple) and len(x) == 1:
                x = x[0]
            m = img_mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(x, mask))
        return out

    def _get_mask(self, nextFeatureMap, beforeMask):
        x = nextFeatureMap
        m = beforeMask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        return mask

    def _pos_embedding_generate(self, xs):
        out: List[NestedTensor] = []
        pos = []
        for x in xs:
            out.append(x)
            # position encoding
            pos.append(self.pos_in(x))  # .to(x.tensors.dtype))
        return out, pos

    def forward(self, vision_feat, text_feat, img_mask, word_mask, **kwargs):
        word_feature, sentence_feature = text_feat[-1]
        if isinstance(word_feature, list):
            word_feature = sum(word_feature) / len(word_feature)
        fv = self._masks_generate(vision_feat, img_mask)
        features, pos = self._pos_embedding_generate(fv)
        feature_last, mask_last = features[3].decompose()
        bs, c, h, w = feature_last.shape
        conv6_1 = self.conv6_1(feature_last)
        conv6_2 = self.conv6_2(conv6_1)
        conv7_1 = self.conv7_1(conv6_2)
        conv7_2 = self.conv7_2(conv7_1)
        conv8_1 = self.conv8_1(conv7_2)
        conv8_2 = self.conv8_2(conv8_1)

        conv5 = self.input_proj(feature_last)
        fv1 = conv5.view(bs, 256, -1)
        fv2 = conv6_2.view(bs, 256, -1)
        fv3 = conv7_2.view(bs, 256, -1)
        fv4 = conv8_2.view(bs, 256, -1)
        fv2_mask = self._get_mask(conv6_2, mask_last)
        fv3_mask = self._get_mask(conv7_2, fv2_mask)
        fv4_mask = self._get_mask(conv8_2, fv3_mask)

        pos1 = pos[-1]
        pos2 = self.pos(NestedTensor(conv6_2, fv2_mask)).to(conv6_2.dtype)
        pos3 = self.pos(NestedTensor(conv7_2, fv3_mask)).to(conv7_2.dtype)
        pos4 = self.pos(NestedTensor(conv8_2, fv4_mask)).to(conv8_2.dtype)
        fvpos1 = pos1.view(bs, 256, -1)
        fvpos2 = pos2.view(bs, 256, -1)
        fvpos3 = pos3.view(bs, 256, -1)
        fvpos4 = pos4.view(bs, 256, -1)

        fv = torch.cat((fv1, fv2), dim=2)
        fv = torch.cat((fv, fv3), dim=2)
        fv = torch.cat((fv, fv4), dim=2)
        fv = fv.permute(2, 0, 1)
        textFeature = torch.cat([word_feature, sentence_feature.unsqueeze(1)], dim=1)
        fl = self.l_proj(textFeature)
        fl = fl.permute(1, 0, 2)
        fvl = torch.cat((fv, fl), dim=0)

        word_mask = word_mask.to(torch.bool)
        word_mask = ~word_mask
        sentence_mask = torch.zeros((bs, 1)).to(word_mask.device).to(torch.bool)
        text_mask = torch.cat((word_mask, sentence_mask), dim=1)
        vis_mask = torch.cat((mask_last.view(bs, -1), fv2_mask.view(bs, -1)), dim=1)
        vis_mask = torch.cat((vis_mask, fv3_mask.view(bs, -1)), dim=1)
        vis_mask = torch.cat((vis_mask, fv4_mask.view(bs, -1)), dim=1)
        fvl_mask = torch.cat((vis_mask, text_mask), dim=1)

        flpos = self.text_pos_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        fvpos = torch.cat((fvpos1, fvpos2), dim=2)
        fvpos = torch.cat((fvpos, fvpos3), dim=2)
        fvpos = torch.cat((fvpos, fvpos4), dim=2)
        fvpos = fvpos.permute(2, 0, 1)
        fvlpos = torch.cat((fvpos, flpos), dim=0)

        out_layers = self.DE(fv1.permute(2, 0, 1), fvl, fvl_mask, fvlpos, fvpos1.permute(2, 0, 1))
        fv1_encode = out_layers[-1].permute(1, 2, 0)

        refineFeature = fv1_encode.view(bs, 256, h, w)
        out = self.transformer(refineFeature, mask_last, pos1)
        return out, word_feature

    def init_weights(self):
        super(CNNMGVLFBridge, self).init_weights()
        if self.init_cfg is not None:
            if isinstance(self.init_cfg, dict) and self.init_cfg["type"] == "Pretrained":
                dict_trained = torch.load(self.init_cfg["checkpoint"])['model']
                dict_new = OrderedDict(self.state_dict())
                for key in dict_new.keys():
                    if key in dict_trained.keys():
                        dict_new[key] = dict_trained[key]
                self.load_state_dict(dict_new, False)


class Transformer(nn.Module):

    def __init__(self, d_model=256, nhead=8, num_encoder_layers=6, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False):
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
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        mask = mask.flatten(1)
        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


def generate_position_embedding(hidden_dim, position_embedding):
    N_steps = hidden_dim // 2
    if position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif position_embedding in ('v3', 'learned'):
        N_steps = 256
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {position_embedding}")
    return position_embedding


class Transformer_Decoder(nn.Module):
    def __init__(self, d_model=512, nhead=8,
                 num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, tgt, memory, mask, pos_embed, query_embed):
        hs = self.decoder(tgt, memory, memory_key_padding_mask=mask, pos=pos_embed, query_pos=query_embed)
        return hs
