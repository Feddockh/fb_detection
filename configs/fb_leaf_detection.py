# Set the base configuration file for Faster R-CNN with ResNet-50 backbone and FPN neck
_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

# Specify the dataset type and classes
dataset_type = 'CocoDataset'
classes = ('Apple_Normal', 'Apple_Fruit_burn', 'Other')

# Specify the data root (data location)
data_root = '/media/hayden/Extreme SSD/korea_dataset_reorganized/'

# Dataset configurations
train_dataset = dict(
    type='CocoDataset',
    ann_file=data_root + 'train/REleaf_train.json',
    data_prefix=dict(img=data_root + 'train/leaf/'),
    metainfo=dict(classes=classes),
)
val_dataset = dict(
    type='CocoDataset',
    ann_file=data_root + 'val/REleaf_val.json',
    data_prefix=dict(img=data_root + 'val/leaf/'),
    metainfo=dict(classes=classes),
)

# Dataloader configurations
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=train_dataset,
    val=val_dataset,
    test=val_dataset,  # Use validation for testing
)
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=train_dataset,
    sampler=dict(type='DefaultSampler', shuffle=True),
)
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    dataset=val_dataset,
    sampler=dict(type='DefaultSampler', shuffle=False),
)

# Model modifications to fit your class labels
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes),
        )
    )
)

# Optimizer and training settings
optimizer = dict(type='SGD', lr=0.001, momentum=0.9, weight_decay=0.0001)
optim_wrapper = dict(type='AmpOptimWrapper', optimizer=optimizer)

runner = dict(type='EpochBasedRunner', max_epochs=12)

# Checkpoint saving path
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='auto',
        backend_args=dict(backend='local')  # Ensure the backend is properly set
    ),
    logger=dict(type='LoggerHook', interval=50)  # Log every 50 iterations
)

# Visualization configuration
visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=[dict(type='TensorboardVisBackend')],
    name='visualizer'
)

# Evaluation metrics
val_evaluator = dict(
    type='CocoMetric',
    ann_file=val_dataset['ann_file'],
    metric='bbox',
)