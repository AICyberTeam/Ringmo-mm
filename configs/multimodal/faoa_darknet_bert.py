_base_ = [
    '../_base_/datasets/rsvg.py',
    '../_base_/schedules/schedule_6x.py',
    '../_base_/default_runtime.py'
]
find_unused_parameters=True

angle_version = 'le90'
model = dict(
    type='VisualGround',
    multimodal_backbone=dict(
        type="SimpleMultiModal",
        backbone_left=dict(
        type='Darknet',
        depth=53,
        out_indices=(3, 4, 5),
       # init_cfg=dict(type='Pretrained', checkpoint='open-mmlab://darknet53'),
    ),
        backbone_right=dict(
            type="Bert",
            init_cfg=dict(
                bert_model="/mnt/sdb/share1416/airalgorithm/code/ringmoMultiModal/ringmouscframework/saved_models/bert-base-uncased.tar.gz",
                tuned=True
            )
        ),
        # bridges=[
        #     dict(type="CNNMGVLFBridge",
        #          hidden_dim=256,
        #          dropout=0.1,
        #          nheads=8,
        #          dim_feedforward=2048,
        #          enc_layers=6,
        #          pre_norm=True,
        #          position_embedding_category='sine',
        #          in_channel=2048,
        #          init_cfg=dict(type='Pretrained',
        #                        checkpoint='/mnt/sdb/share1416/airalgorithm/pretrained/detr-r50-e632da11.pth')
        #          ),
        # ],
        top_bridge=dict(
            type="FAOATopBridge",
            hidden_dim=256,
            dropout=0.1,
            leaky=False,
            coordmap=True,
            language_channel=768,
            vision_channels=(256, 512, 1024)
        ),
    ),
    neck=dict(
            type='YOLOV3Neck',
            num_scales=3,
            in_channels=[520, 520, 520],
            out_channels=[96, 96, 96]
        ),
    bbox_head=dict(
        type='VGYOLOV3Head',
        in_channels=[96, 96, 96],
        out_channels=[96, 96, 96],
        anchor_generator=dict(
            type='YOLOAnchorGenerator',
            base_sizes=[[(351,357), (123,126), (552,535)],
                        [(602,236), (41,41), (357,134)],
                        [(193,230), (327,608), (154,473)]],
            strides=[32, 16, 8]),
        bbox_coder=dict(type='YOLOBBoxCoder'),
        featmap_strides=[32, 16, 8],
        loss_conf=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=1.0,
            reduction='sum'),
        loss_xy=dict(
            type='CrossEntropyLoss',
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='sum'),
        loss_wh=dict(type='MSELoss', loss_weight=2.0, reduction='sum')),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(
            type='GridAssigner',
            pos_iou_thr=0.5,
            neg_iou_thr=0.5,
            min_pos_iou=0)),
    test_cfg=dict(
        nms_pre=1,
        min_bbox_size=0,
        score_thr=0.,
        conf_thr=0.005,
        nms=dict(type='nms', iou_threshold=0.45),
        max_per_img=100)
)
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-5,
                 paramwise_cfg=dict(custom_keys={
                     'backbone_model_right': dict(lr_mult=0.1),
                     'transformer':dict(lr_mult=0.1)

                 }))

evaluation = dict(interval=1, metric='detAcc@0.5')
runner = dict(type='EpochBasedRunner', max_epochs=300)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[60, 90])
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=10,
)


