_base_ = [
    '../_base_/datasets/loveda.py', '../_base_/models/aerialformer.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule.py'
]

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_tiny_patch4_window7_224_20220317-1cdeb081.pth'  # noqa


model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    test_cfg=dict(mode='slide', crop_size=(512, 512), stride=(64, 64)))

# optimizer
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.00006,
    betas=(0.9, 0.999),
    weight_decay=0.01,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))

lr_config = dict(
    _delete_=True,
    policy='poly',
    warmup='linear',
    warmup_iters=1500,
    warmup_ratio=1e-6,
    power=1.0,
    min_lr=0.0,
    by_epoch=False)


# data = dict(samples_per_gpu=1, workers_per_gpu=1) # 1 GPU x 8 samples/gpu = 8 batch size
# runner = dict(type='IterBasedRunner', max_iters=800000)
# checkpoint_config = dict(by_epoch=False, interval=50)

data = dict(samples_per_gpu=8, workers_per_gpu=8) # 1 GPU x 8 samples/gpu = 8 batch size

# Activete the following lines to create the results for the test set
# data = dict(
#     test=dict(
#         img_dir='img_dir/test',
        # ann_dir='ann_dir/NA'))