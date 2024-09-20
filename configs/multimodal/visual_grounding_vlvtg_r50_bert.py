from configs._base_.models.components.backbones import ringmo1b, resnet50
from configs._base_.models.components.bbox_heads import roi_trans_head

_base_ = [
    '../_base_/datasets/rsvg.py',
    '../_base_/schedules/schedule_6x.py',
    '../_base_/default_runtime.py'
]
# find_unused_parameters=True

angle_version = 'le90'
model = dict(
    type='VisualGround',
    multimodal_backbone=dict(
        type="SimpleMultiModal",
        backbone_left={**resnet50['backbone'], "out_indices": [0, 1, 2, 3]},
        backbone_right=dict(
            type="Bert",
            init_cfg=dict(
                bert_model="/mnt/sdb/share1416/airalgorithm/code/ringmoMultiModal/ringmouscframework/saved_models/bert-base-uncased.tar.gz",
                tuned=True
            )
        ),
        top_bridge=dict(
            type="VLTVGBridge",
            num_queries=1,
            query_dim=256,
            norm_dim=256,
            num_layers=6,
            num_extra_layers=1,
        )

    ),
    bbox_head=dict(
        type='VLTVGHead',
        hidden_dim=256,
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
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
optimizer = dict(type='AdamW', lr=5e-5, weight_decay=1e-4,
                 paramwise_cfg=dict(custom_keys={'backbone_model_right': dict(lr_mult=0.1)
                                                # 'backbone_model_left': dict(lr_mult=0.1),
                                                # 'trans_encoder':dict(lr_mult=0.1)
                                                 }))

evaluation = dict(interval=1, metric='detAcc@0.5')
runner = dict(type='EpochBasedRunner', max_epochs=300)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[150, 250])
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=10,
)
