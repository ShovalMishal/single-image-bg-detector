dataloader = dict(
    batch_size=20,
    # num_workers=2,
    num_workers=0,
    # persistent_workers=True,
    persistent_workers=False,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='DOTAv2Dataset',
        data_root='/home/shoval/Documents/Repositories/data/gsd_normalized_dataset',
        ann_file='train/annfiles/',
        data_prefix=dict(img_path='train/images/'),
        test_mode=True,
        pipeline=[
            dict(type='mmdet.LoadImageFromFile', backend_args=None),
            dict(type='mmdet.Resize', scale=(1024, 1024), keep_ratio=True),
            dict(
                type='mmdet.LoadAnnotations', with_bbox=True, box_type='qbox'),
            dict(
                type='ConvertBoxType',
                box_type_mapping=dict(gt_bboxes='rbox')),
            dict(
                type='mmdet.PackDetInputs',
                meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                           'scale_factor'))
        ]))
