ringmo1b = dict(
    backbone=dict(
        # _delete_=True,
        type='RingMoGiant',
        in_channels=3,
        embed_dims=448,
        depths=[2, 2, 18, 2],
        num_heads=[14, 28, 56, 112],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained',
                      checkpoint='/mnt/sdb/share1416/airalgorithm/swinv2_giant_22k_500k.pth')
    ),
    neck=dict(
        type='FPN',
        in_channels=[448, 896, 1792, 3584],
        out_channels=256,
        num_outs=5),
)

resnet50 = dict(
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        zero_init_residual=False,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained',
                      checkpoint='/mnt/sdb/share1416/airalgorithm/resnet50-0676ba61.pth')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',  # use P5
        num_outs=5,
        relu_before_extra_convs=True),
)



swin_mix = dict(
    backbone=dict(
        type='SwinTransformerMix',
        init_cfg=dict(type='Pretrained',
                      checkpoint="./checkpoints/mmseg_swin_lite.pth"),
        pretrain_img_size=224,
        embed_dims=96,
        patch_size=4,
        window_size=7,
        mlp_ratio=4,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        strides=(4, 2, 2, 2),
        out_indices=(0, 1, 2, 3),
        qkv_bias=True,
        qk_scale=None,
        patch_norm=True,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        use_abs_pos_embed=False,
        act_cfg=dict(type='GELU'),
        norm_cfg=dict(type='LN', requires_grad=True)),
    neck=dict(
        type='FPN',
        in_channels=[96, 192, 384, 768],
        out_channels=256,
        num_outs=5),
)
