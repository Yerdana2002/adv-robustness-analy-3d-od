voxel_size = [0.2, 0.2, 0.4]

model = dict(
    type='CenterPoint_f',
    pts_voxel_layer=dict(
        max_num_points=5,
        voxel_size=voxel_size,
        max_voxels=(8000, 20000)),
    pts_voxel_encoder=dict(type='HardSimpleVFE', num_features=4),
    pts_middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[16, 748, 748],  # xy dims halved from original
        output_channels=128,
        order=('conv', 'norm', 'act'),
        encoder_channels=((16, 16, 32),
                          (32, 32, 64),
                          (64, 64, 128),
                          (128, 128)),
        encoder_paddings=((0, 0, 1),
                          (0, 0, 1),
                          (0, 0, [0, 1, 1]),
                          (0, 0)),
        block_type='basicblock'),
    pts_backbone=dict(
        type='SECOND',
        in_channels=256,
        out_channels=[128, 256],
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        conv_cfg=dict(type='Conv2d', bias=False)),
    pts_neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        out_channels=[256, 256],
        upsample_strides=[1, 2],
        norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
        upsample_cfg=dict(type='deconv', bias=False, output_padding=0),
        use_conv_for_no_stride=True),
    pts_bbox_head=dict(
        type='CenterHead_f',
        in_channels=sum([256, 256]),
        tasks=[
            dict(num_class=1, class_names=['Car']),
            dict(num_class=1, class_names=['Pedestrian']),
            dict(num_class=1, class_names=['Cyclist']),
        ],
        common_heads=dict(reg=(2, 2),
                          height=(1, 2),
                          dim=(3, 2),
                          rot=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            post_center_range=[-10, -50, -10, 80.4, 50, 10],
            max_num=100,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],  # still uses XY plane for BEV
            code_size=7),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            grid_size=[748, 748, 15],  # xy halved, z same as original (since z voxel=0.4 vs 0.2)
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0] * 8)),
    test_cfg=dict(
        pts=dict(
            post_center_limit_range=[-10, -50, -10, 80.4, 50, 10],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            nms_type='rotate',
            pre_max_size=4096,
            post_max_size=512,
            nms_thr=0.2)))
