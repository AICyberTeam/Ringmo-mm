# Copyright (c) OpenMMLab. All rights reserved.
import warnings

from mmdet.models.builder import MODELS

MULTIMODAL = MODELS
BRIDGE = MODELS
MULTIMODALDETECTOR = MODELS
BACKBONE = MODELS
BBOX_HEAD = MODELS
TRANSFORMER = MODELS
LOSSES = MODELS
MULTIMODALRETRIEVER = MODELS
SIMILARITY_HEAD = MODELS


def build_backbone(cfg):
    """Build backbone."""
    if not cfg:
        return None
    return BACKBONE.build(cfg)


def build_multimodal(cfg):
    return MULTIMODAL.build(cfg)


def build_bridge(cfg):
    return BRIDGE.build(cfg)


def build_similarity_head(cfg):
    return SIMILARITY_HEAD


def build_loss(cfg):
    """Build loss."""
    return LOSSES.build(cfg)


def build_multimodal_detector(cfg, train_cfg=None, test_cfg=None):
    """Build detector."""
    if train_cfg is not None or test_cfg is not None:
        warnings.warn(
            'train_cfg and test_cfg is deprecated, '
            'please specify them in model', UserWarning)
    assert cfg.get('train_cfg') is None or train_cfg is None, \
        'train_cfg specified in both outer field and model field '
    assert cfg.get('test_cfg') is None or test_cfg is None, \
        'test_cfg specified in both outer field and model field '
    return MULTIMODALDETECTOR.build(
        cfg, default_args=dict(train_cfg=train_cfg, test_cfg=test_cfg))


def build_multimodal_retriever(cfg):
    return MULTIMODALRETRIEVER.build(cfg)
