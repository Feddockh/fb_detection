import os
from mmengine.config import Config
from mmengine.runner import Runner

# Register all modules in mmdetection
from mmdet.utils import register_all_modules
register_all_modules()

# Object for detection
obj = 'branch'

# Load the configuration file
config_file = f'./configs/fb_{obj}_detection.py'
cfg = Config.fromfile(config_file)

# Set the working directory
cfg.work_dir = f'./{obj}_models'
os.makedirs(cfg.work_dir, exist_ok=True)

# Specify the checkpoint file to evaluate
checkpoint_file = os.path.join(cfg.work_dir, 'best_coco_bbox_mAP_epoch_12.pth')

# Ensure the checkpoint file exists
if not os.path.exists(checkpoint_file):
    raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_file}')

# Use the validation dataset as the test dataset
cfg.test_dataloader = cfg.val_dataloader
cfg.test_evaluator = cfg.val_evaluator

# Remove training-specific configurations
cfg.pop('train_dataloader', None)
cfg.pop('optim_wrapper', None)
cfg.pop('param_scheduler', None)
cfg.pop('train_cfg', None)
cfg.pop('default_hooks', None)

# Set the checkpoint to load
cfg.load_from = checkpoint_file

# Initialize the runner
runner = Runner.from_cfg(cfg)

# Run evaluation
eval_results = runner.test()

# Print evaluation results
print('\nEvaluation Results:')
for k, v in eval_results.items():
    print(f'{k}: {v}')
