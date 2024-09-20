from configs._base_.models.components.backbones import ringmo1b

_base_ = [
    '../_base_/datasets/rsvg.py',
    '../_base_/default_runtime.py'
]
find_unused_parameters=True
angle_version = 'le90'
model = dict(
    type='VisualGround',
    multimodal_backbone=dict(
        type="SimpleMultiModal",
        backbone_left={**ringmo1b['backbone'], "out_indices": [0,1,2,3]},
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
        #          dim_feedforward=1024,
        #          enc_layers=6,
        #          pre_norm=True,
        #          position_embedding_category='sine',
        #          in_channel=3584,
        #          # init_cfg=dict(type='Pretrained',
        #          #               checkpoint='/mnt/sdb/share1416/airalgorithm/pretrained/detr-r50-e632da11.pth')
        #          ),
        #
        # ],
        # top_bridge=dict(
        #     type="VLFBridge",
        #     hidden_dim=128,
        #     dropout=0.1,
        #     nheads=8,
        #     dim_feedforward=1024,
        #     enc_layers=6,
        #     dec_layers=6,
        #     pre_norm=True,
        #     position_embedding_category='learned',
        #     N_steps=128,
        #     vision_channel=3584,
        #     language_channel=768,
        #
        # ),
        top_bridge=dict(
            type="VLFBridge",
            hidden_dim=256,
            dropout=0.1,
            nheads=8,
            dim_feedforward=1024,
            enc_layers=6,
            dec_layers=6,
            pre_norm=True,
            position_embedding_category='learned',
            N_steps=256,
            vision_channel=3584,
            language_channel=768,
            # init_cfg=dict(type='Pretrained',
            #               checkpoint='/mnt/sdb/share1416/airalgorithm/pretrained/detr_transformer.pth'),
        )
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
        loss_bbox=dict(type='L1Loss', loss_weight=5.0),
        loss_iou=dict(type='GIoULoss', loss_weight=2.0),
    ),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict()
)
evaluation = dict(interval=1, metric='detAcc@0.5')
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-5,
                 paramwise_cfg=dict(custom_keys={#'norm': dict(decay_mult=0.),
                                                 'backbone_model_right': dict(lr_mult=0.1)
                                                 }))

lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[200])

optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))

data = dict(
    samples_per_gpu=4,
    workers_per_gpu=5, )

runner = dict(type='EpochBasedRunner', max_epochs=300)

checkpoint_config = dict(interval=1)
log_config = dict(
    interval=400,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])

