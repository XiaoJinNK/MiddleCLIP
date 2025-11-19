_base_ = './base_config.py'

# model settings
model = dict(
    name_path='./configs/cls_coco_stuff.txt',
    temp_thd=0.11,
    delete_same_entity=True,
    attn_rcs_weights=[2.0, 0.4],
    attn_sfr_weights=[1.8, 0.7],
    shallow_layer_idx=8,
    shallow_fusion_weight=0.5,
    branch_fusion_weight=0.3,
    mid_layer=[10],
    branch_layers_from_end=3, 
)

# dataset settings
dataset_type = 'COCOStuffDataset'
data_root = './data/coco_stuff164k'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2048, 336), keep_ratio=True),
    dict(type='LoadAnnotations'),
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
            img_path='images/val2017', seg_map_path='annotations/val2017'),
        pipeline=test_pipeline))
