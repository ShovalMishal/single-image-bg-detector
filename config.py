dataloader = dict(
    batch_size=1,
    # num_workers=2,
    num_workers=0,
    # persistent_workers=True,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTAv2Dataset',
        data_root='/home/shoval/Documents/Repositories/data/gsd_normalized_dataset',
        ann_file='train/labelTxt/',
        data_prefix=dict(img_path='train/images/'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile'),
            dict(
                type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='hbox')),
            dict(type='mmdet.Resize', scale=(512, 512), keep_ratio=True),
            dict(type='Normalize', mean=(0, 0, 0), std=(255.0, 255.0, 255.0), to_rgb=False),
            dict(type='Normalize', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
        ]))

patches_assigner = dict(
    type='mmdet.MaxIoUAssigner',
    pos_iou_thr=0.5,
    neg_iou_thr=0.1,
    min_pos_iou=0.5,
    match_low_quality=True,
    ignore_iof_thr=-1)
