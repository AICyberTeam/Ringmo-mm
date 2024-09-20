# dataset settings
dataset_type = 'PairwiseDataset'
data_root = '/mnt/sdb/share1416/airalgorithm/datasets/hvsa/'
vocab_dictionary_path = data_root + 'rsicd_splits_vocab.json'
norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='BertTextTokenize', query_len=40,
         bert_model_path='/mnt/sdb/share1416/airalgorithm/code/ringmoMultiModal/ringmouscframework/saved_models'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(640, 640), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **norm_cfg),
    #dict(type='RandomAffine', skip_filter=False),
    dict(type='Pad', size_divisor=32),
    dict(type="VoidMask"),
    dict(type='DefaultFormatBundle'),
    dict(
        type="MultiModalCollect",
        text_keys=('word_id', 'word_mask'),
        gt_keys=('gt_bboxes', 'gt_labels'),
        appendix_keys=('word_mask','img_metas', 'img_mask'), #'parse_out'),#, ),
        img_data_keys=('img',),
        img_meta_keys=('filename', 'ori_filename', 'ori_shape',
                       'img_shape', 'pad_shape', 'scale_factor', 'flip',
                       'flip_direction', 'img_norm_cfg')
    )
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiModalMultiScaleFlipAug',
        img_scale=(640, 640),
        flip=False,
        transforms=[
            dict(type='BertTextTokenize', query_len=40,
                 bert_model_path='/mnt/sdb/share1416/airalgorithm/code/ringmoMultiModal/ringmouscframework/saved_models'),
            dict(type='Resize'),
            dict(type='Normalize', **norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type="VoidMask"),
            dict(type='DefaultFormatBundle'),
            dict(
                type="MultiModalCollect",
                text_keys=('word_id', 'word_mask'),
                appendix_keys=('word_mask','img_metas', 'img_mask'),
                img_data_keys=('img', ),
                img_meta_keys=('filename', 'ori_filename', 'ori_shape',
                               'img_shape', 'pad_shape', 'scale_factor', 'flip',
                               'flip_direction', 'img_norm_cfg')
            )
        ])
]
classes = ['vehicle', 'overpass', 'chimney', 'Expressway-Service-area', 'harbor', 'basketballcourt',
           'baseballfield', 'groundtrackfield', 'bridge', 'stadium', 'windmill', 'airplane', 'tenniscourt',
           'airport', 'ship', 'trainstation', 'storagetank', 'golffield', 'dam', 'Expressway-toll-station']
data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'train_img_cap_match.txt',
        image_path=data_root + 'images/',
        dictionary_path = vocab_dictionary_path
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations',
        classes=classes,
        img_prefix=data_root + 'JPEGImages',
        image_suffix='.jpg',
        split_file=data_root + 'val.txt',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'Annotations',
        classes=classes,
        img_prefix=data_root + 'JPEGImages',
        image_suffix='.jpg',
        split_file=data_root + 'val.txt',
        pipeline=test_pipeline)
)
