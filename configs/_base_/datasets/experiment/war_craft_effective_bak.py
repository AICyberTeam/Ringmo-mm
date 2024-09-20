# dataset settings
dataset_type = 'AircasDataset'

data_root_negative_2 = '/mnt/sdb/share1416/airalgorithm/datasets/warcraft-gf-jun202311-version2/effective/'
# data_root_version4 = '/mnt/sdb/share1416/airalgorithm/datasets/ccd_warcraft_GF2_version4/'
data_root_negative_1 = '/mnt/sdb/share1416/airalgorithm/datasets/warcraft-gf-jun202311-version1/effective/'
data_root_version5 = '/mnt/sdb/share1416/airalgorithm/datasets/ccd_warcraft_GF2_version5/'

data_root_list = [
    data_root_negative_1]  ## [data_root_negative_1] + [data_root_version5] * 2 + [data_root_negative_2] * 3
img_norm_cfg = dict(
    mean=[122.52188731, 122.52188731, 122.52188731], std=[63.545590175277034, 63.545590175277034, 63.545590175277034],
    to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RResize', img_scale=(100, 100)),
    dict(type='RRandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(100, 100),
        flip=False,
        transforms=[
            dict(type='RResize'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img'])
        ])
]
classes = ['两栖舰', '其它', '巡洋舰', '护卫舰', '未知军舰', '潜艇', '航空母舰', '驱逐舰']  # ['潜艇', '驱逐舰', '巡洋舰', '护卫舰', '其它', '航空母舰', '两栖舰']

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=[dict(
        type=dataset_type,
        ann_file=data_root + 'train/xmls/',
        classes=classes,
        img_prefix=data_root + 'train/imgs/',
        image_suffix='.png',
        filter_empty_gt=False,
        pipeline=train_pipeline) for i, data_root in enumerate(data_root_list)
    ],
    val=dict(
        type=dataset_type,
        ann_file=data_root_version5 + 'val/xmls/',
        classes=classes,
        img_prefix=data_root_version5 + 'val/imgs/',
        image_suffix='.png',
        filter_empty_gt=False,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root_version5 + 'val/xmls/',
        img_prefix=data_root_version5 + 'val/imgs/',
        pipeline=test_pipeline))
