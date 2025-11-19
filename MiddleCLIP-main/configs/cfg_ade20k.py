_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_ade20k.txt',
    temp_thd=0.25,
    delete_same_entity=True,
    attn_rcs_weights=[2.0, 0.3], 
    attn_sfr_weights=[2.1, 0.7],
    shallow_layer_idx=8,
    shallow_fusion_weight=0.4,
    branch_fusion_weight=0.4,
    mid_layer=[10],
    branch_layers_from_end=3, 
)

# dataset settings
dataset_type = 'ADE20KDataset'
data_root = './data/ADE20K'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]

test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='images/validation',
            seg_map_path='annotations/validation'),
        pipeline=test_pipeline))
