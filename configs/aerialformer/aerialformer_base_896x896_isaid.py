_base_ = ["./aerialformer_tiny_896x896_isaid.py"]
checkpoint_file = "https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth"  # noqa
decoder_norm_cfg = dict(type='SyncBN', requires_grad=True)

model = dict(
    backbone=dict(
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file),
        pretrain_img_size=384,
        embed_dims=128,
        window_size=12,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        conv_norm_cfg=decoder_norm_cfg
        ),
    decode_head=dict(
        in_channels=[64, 128, 256, 512, 1024],
        channels=128,
        norm_cfg=decoder_norm_cfg,
    )
)

data = dict(samples_per_gpu=4, workers_per_gpu=4) # 2 GPUs x 4 samples/gpu = 8 batch size
