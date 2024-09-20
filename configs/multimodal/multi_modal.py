from configs._base_.models.components.backbones import ringmo1b
from configs._base_.models.components.bbox_heads import roi_trans_head

_base_ = [
    '../_base_/datasets/rsvg.py',
    '../_base_/schedules/schedule_6x.py',
    '../_base_/default_runtime.py'
]
find_unused_parameters=True
norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
angle_version = 'le90'
model = dict(
    type='MultiModalYoloV3',
    multimodal_backbone=dict(
        type="SimpleMultiModal",
        backbone_left={**ringmo1b['backbone'], "out_indices": [3]},
        backbone_right=dict(
            type="Bert",
            init_cfg=dict(
                bert_model="/mnt/sdb/share1416/airalgorithm/code/ringmoMultiModal/ringmouscframework/saved_models/bert-base-uncased.tar.gz",
                tuned=True
            )
        ),
        top_bridge=dict(
            type="VLFBridge",
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            dim_feedforward=2048,
            enc_layers=6,
            dec_layers=6,
            pre_norm=True,
            position_embedding_category='learned',
            N_steps=256
        )
        # bridges=[
        #             dict(type="IdentityBridgeBase",)
        #         ] * 5,
    ),
    # neck=dict(
    #     type='FPN',
    #     in_channels=[256],
    #     out_channels=256,
    #     start_level=1,
    #     add_extra_convs='on_input',
    #     num_outs=5),
    bbox_head=dict(
        type='MGVLFHead',
        num_classes=20,
        in_channels=256,
    ),
    # model training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='MaxIoUAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1),
        sampler=dict(
            type='PseudoSampler'),  # Focal loss should use PseudoSampler
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.05,
        # score_thr=0.5,
        nms=dict(type='nms', iou_threshold=0.5),
        max_per_img=100)
)
optimizer = dict(_delete_=True, type='AdamW', lr=1e-4, betas=(0.9, 0.999), weight_decay=1e-4,
                 paramwise_cfg=dict(custom_keys={'absolute_pos_embed': dict(decay_mult=0.),
                                                 'relative_position_bias_table': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.)}))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=5, )
