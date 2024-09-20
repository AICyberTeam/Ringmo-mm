from ...builder import build_backbone, build_bridge
from abc import ABCMeta
from mmcv.runner import BaseModule
from ringmomultimodal.models.backbones.utils.model_flow import ModelFlow, cond_run
import torch
import warnings


class RingMoMultiModalBase(BaseModule, metaclass=ABCMeta):
    def __init__(self, backbone_left, backbone_right, bridges=None, init_cfg=None, top_bridge=None):
        BaseModule.__init__(self, init_cfg=init_cfg)
        self.fp16_enabled = False

        if bridges is None:
            bridges = []
        assert isinstance(bridges, list)
        self.backbone_model_left = build_backbone(backbone_left)
        self.backbone_model_right = build_backbone(backbone_right)
        self.backbone_left_flow = ModelFlow(self.backbone_model_left)
        self.backbone_right_flow = ModelFlow(self.backbone_model_right)
        self.bridges = torch.nn.ModuleList()
        self.unspecified = True
        self.bridge_info = bridges
        for bridge_info in bridges:
            if {"left_in_feat", "right_in_feat", "left_out_feat", "right_out_feat"}.intersection(
                    set(bridge_info.keys())):
                self.unspecified = False
            else:
                assert self.unspecified, \
                    'Options "left_in_feat", "right_in_feat", "left_out_feat", "right_out_feat" should be set or not set in all bridges'
            assert not ('left_in_feat' in bridge_info) ^ ('right_in_feat' in bridge_info), \
                "The option 'left_in_feat' and 'right_in_feat' should be specified both"
            assert not ('left_out_feat' in bridge_info) ^ ('right_out_feat' in bridge_info), \
                "The option 'left_out_feat' and 'right_out_feat' should be specified both"
            bridge_model = build_bridge(bridge_info)
            self.bridges.append(bridge_model)
        self.bridge_first = False
        if self.bridges:
            if not len(self.backbone_left_flow) == len(self.backbone_right_flow) == len(
                    self.bridges):
                warnings.warn("The stage num of two models and bridges should be same.")
                self.bridge_first = True
        if top_bridge is None:
            top_bridge = {'type': 'LeftOutTopBridgeBase'}
        self.top_bridge = build_bridge(top_bridge)

    def backbone_unspecified(self, input_left, input_right, **kwargs):
        feat_lefts, feat_rights = [], []
        if len(self.bridges) > 1:
            for bridge, stage_left, stage_right in zip(self.bridges, self.backbone_left_flow, self.backbone_right_flow):
                feat_left, feat_right = input_left, input_right
                feat_left = cond_run(stage_left, feat_left)
                feat_right = cond_run(stage_right, feat_right)
                feat_lefts.append(feat_left)
                feat_rights.append(feat_right)
                feat_left, feat_right = bridge(feat_lefts, feat_rights, **kwargs)
                input_left, input_right = feat_left, feat_right
            if len(self.bridges) < len(self.backbone_left_flow):
                left_output, right_output = self.backbone_left_flow.outputs, self.backbone_right_flow.outputs
            else:
                left_output, right_output = feat_lefts[-1], feat_rights[-1]
        else:
            left_output = cond_run(self.backbone_left_flow, input_left)
            right_output = cond_run(self.backbone_right_flow, input_right)
            if len(self.bridges) == 1:
                left_output, right_output = self.bridges[-1](left_output, right_output, **kwargs)
        output = self.top_bridge(left_output, right_output, **kwargs)
        return output

    def backbone_specified(self, input_left, input_right):
        raise NotImplementedError

    def forward(self, input_left, input_right, **kwargs):
        if self.unspecified:
            output = self.backbone_unspecified(input_left, input_right, **kwargs)
        else:
            raise NotImplementedError
            output = self.backbone_specified(input_left, input_right)
        return output
