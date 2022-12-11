import os
import sys
import random
import matplotlib.pyplot as plt
import glob
from mrcnn import visualize
import mrcnn.model as modellib
from mrcnn.model import log
from keras.models import load_model

from tilesTraining import MahjongTileConfig, MahjongTileDataset

# Root directory of the project
ROOT_DIR = os.getcwd()

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library

custom_WEIGHTS_PATH = sorted(glob.glob("/logs/*/mask_rcnn_*.h5"))[-1]

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

config = MahjongTileConfig()
custom_DIR = os.path.join(ROOT_DIR, "dataset")

class InferenceConfig(config.__class__):
	# Run detection on one image at a time
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Device to load the neural network on.
# Useful if you're training a model on the same 
# machine, in which case use CPU and leave the
# GPU for training.
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"

def get_ax(rows=1, cols=1, size=16):
	"""Return a Matplotlib Axes array to be used in
	all visualizations in the notebook. Provide a
	central point to control graph sizes.
	
	Adjust the size attribute to control how big to render images
	"""
	_, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
	return ax
  
# Load validation dataset
dataset = MahjongTileDataset()
dataset.load_mahjongTile(custom_DIR, "val")

# Must call before using the dataset
dataset.prepare()

print("Images: {}\nClasses: {}".format(len(dataset.image_ids), dataset.class_names))

# Create model in inference mode
model = load_model("mask_rcnn_mahjongtile_0005.h5")

if model == None:
	print("model is none")
	exit(1)

image_id = random.choice(dataset.image_ids)
# for image_id in dataset.image_ids:

image, image_meta, gt_class_id, gt_bbox, gt_mask =\
	modellib.load_image_gt(dataset, config, image_id)
info = dataset.image_info[image_id]
print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id, 
										dataset.image_reference(image_id)))

visualize.display_instances(image, gt_bbox, gt_mask, gt_class_id, 
	dataset.class_names, title="GT")

# Run object detection
results = model.predict(image)

# Display results
ax = get_ax(1)
r = results
visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
							dataset.class_names, r['scores'], ax=ax,
							title="Predictions")
log("gt_class_id", gt_class_id)
log("gt_bbox", gt_bbox)
log("gt_mask", gt_mask)