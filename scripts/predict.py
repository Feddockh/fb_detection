import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from mmengine.config import Config

# Register all modules in MMDetection
from mmdet.utils import register_all_modules
register_all_modules()

# MMDetection imports
from mmdet.registry import DATASETS
from mmdet.apis import init_detector, inference_detector

# Object for detection
obj = 'branch'

# Load the configuration file
config_file = f'./configs/fb_{obj}_detection.py'
cfg = Config.fromfile(config_file)

# Set the working directory
cfg.work_dir = f'./{obj}_models'
os.makedirs(cfg.work_dir, exist_ok=True)

# Specify the checkpoint file to load
checkpoint_file = os.path.join(cfg.work_dir, 'best_coco_bbox_mAP_epoch_12.pth')

# Ensure the checkpoint file exists
if not os.path.exists(checkpoint_file):
    raise FileNotFoundError(f'Checkpoint file not found: {checkpoint_file}')

# Initialize the model
model = init_detector(cfg, checkpoint_file, device='cuda:0')  # Change to 'cpu' if you don't have a GPU

# Build the validation dataset
val_dataset = DATASETS.build(cfg.val_dataloader.dataset)

## Branch Examples
# Healthy example
# index = 4365 # Good example
# index = 4420 # Good example
# index = 4439 # Good example
# FB Examples
# index = 4764 # Bad example
# index = 4765 # Bad example
# index = 4807 # Bad example
# index = 4888 # Good example
# index = 4930 # Ok example

# Other Examples
# index = 20 # Good example
# index = 106 # Good example
index = 798 # Good example

## Leaf Examples
# Healthy example
# index = 5553 # Good example
# index = 5550 # Good example
# index = 5322 # Good example
# FB Examples
# index = 7754 # Good example
# index = 7753 # Good example
# index = 7755 # Bad example
# Other Examples
# index = 2 # Good example
# index = 5003 # Good example
# index = 4924

# Get data information for the selected image
data_info = val_dataset.get_data_info(index)
image_path = data_info['img_path']
img = plt.imread(image_path)

# Get class names
class_names = val_dataset.metainfo['classes']

# Run inference on the image
result = inference_detector(model, image_path)

# Process the inference result
score_thr = 0.6  # Confidence threshold for displaying predictions
pred_instances = result.pred_instances
scores = pred_instances.scores.cpu().numpy()
labels = pred_instances.labels.cpu().numpy()
bboxes = pred_instances.bboxes.cpu().numpy()

# Filter predictions by score threshold
inds = scores >= score_thr
filtered_scores = scores[inds]
filtered_labels = labels[inds]
filtered_bboxes = bboxes[inds]

# Create a figure and axes
fig, ax = plt.subplots(1, figsize=(12, 8))

# Display the image
ax.imshow(img)

# Plot ground truth bounding boxes and labels
for target in data_info['instances']:
    x1, y1, x2, y2 = target['bbox']
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    
    label_index = target['bbox_label']
    ax.text(x1, y1 - 5, f'GT: {class_names[label_index]}', color='g', fontsize=12, backgroundcolor='black')

# Plot predicted bounding boxes and labels
for bbox, label, score in zip(filtered_bboxes, filtered_labels, filtered_scores):
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    # Add label text
    ax.text(x1, y2 + 15, f'Pred: {class_names[label]} {score:.2f}', color='r', fontsize=12, backgroundcolor='black')

# Remove axes and show the plot
ax.axis('off')
plt.tight_layout()
plt.show()

# Save the figure to the figs folder
save = False
if save:
    figs_folder = './figs'
    os.makedirs(figs_folder, exist_ok=True)
    fig.savefig(os.path.join(figs_folder, f'{obj}_{index}_detection_result.png'))
