_base_ = ['./aerialformer_tiny_512x512_loveda.py']
checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_small_patch4_window7_224_20220317-7ba6d6dd.pth'  # noqa
decoder_norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        depths=[2, 2, 18, 2],
        conv_norm_cfg=decoder_norm_cfg),
    decode_head=dict(norm_cfg=decoder_norm_cfg)
)

data = dict(samples_per_gpu=4, workers_per_gpu=4) # 2 GPUs x 4 samples/gpu = 8 batch size

# Activete the following lines to create the results for the test set
# data = dict(
#     test=dict(
#         img_dir='img_dir/test',
#         ann_dir='ann_dir/NA'))