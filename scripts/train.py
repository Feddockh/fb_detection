import os
from mmengine.config import Config
from mmengine.runner import Runner
from mmdet.registry import DATASETS


# Object for detection
obj = 'leaf'

# Load the configuration file
config_file = f'./configs/fb_{obj}_detection.py'
cfg = Config.fromfile(config_file)

# Create the working directory if it doesn't exist
os.makedirs(f'./{obj}_models', exist_ok=True)
cfg.work_dir = f'./{obj}_models'


cfg.resume = True

# Initialize and start the runner
runner = Runner.from_cfg(cfg)
runner.train()