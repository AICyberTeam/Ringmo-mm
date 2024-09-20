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
        backbone_left={**resnet50['backbone'], "out_indices": [0, 1, 2, 3], 'type':'ResNetFlow'},
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
        #
        # ],
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
            N_steps=256,
            vision_channel=2048,
            language_channel=768,
            init_cfg=dict(type='Pretrained',
                          checkpoint='/mnt/sdb/share1416/airalgorithm/pretrained/detr_transformer.pth'),
        )

    ),
    bbox_head=dict(
        type='MGVLFHead',
        num_classes=20,
        in_channels=256,
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
optimizer = dict(type='AdamW', lr=1e-4, weight_decay=1e-5,
                 paramwise_cfg=dict(custom_keys={'backbone_model_right': dict(lr_mult=0.1)}))

evaluation = dict(interval=1, metric='detAcc@0.5')
runner = dict(type='EpochBasedRunner', max_epochs=300)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[150, 250])
dataset_type = 'MultiModalRSVGDataset'
data_root = '/mnt/sdb/share1416/Ringmo_VG/MGVLF/RSVG-pytorch-main/RSVG-pytorch-main/DIOR_RSVG/'
data = dict(
    samples_per_gpu=20,
    workers_per_gpu=10,
    train=dict(
        #split_file=data_root + 'train.txt',
        split_file=data_root + 'val.txt'
    )
)

