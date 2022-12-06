"""
Mask R-CNN
Train on the toy MahjongTile dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
	   the command line as such:

	# Train a new model starting from pre-trained COCO weights
	python3 mahjongTile.py train --dataset=/path/to/mahjongTile/dataset --weights=coco

	# Resume training a model that you had trained earlier
	python3 mahjongTile.py train --dataset=/path/to/mahjongTile/dataset --weights=last

	# Train a new model starting from ImageNet weights
	python3 mahjongTile.py train --dataset=/path/to/mahjongTile/dataset --weights=imagenet

	# Apply color splash to an image
	python3 mahjongTile.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

	# Apply color splash to video using the last weights you trained
	python3 mahjongTile.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from PIL import Image, ImageDraw
import pycocotools.mask as mask_util

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

############################################################
#  Configurations
############################################################


class MahjongTileConfig(Config):
	"""Configuration for training on the toy  dataset.
	Derives from the base Config class and overrides some values.
	"""
	# Give the configuration a recognizable name
	NAME = "mahjongTile"

	# We use a GPU with 12GB memory, which can fit two images.
	# Adjust down if you use a smaller GPU.
	IMAGES_PER_GPU = 2

	# Number of classes (including background)
	NUM_CLASSES = 1 + 1  # Background + mahjongTile

	# Number of training steps per epoch
	STEPS_PER_EPOCH = 100

	# Skip detections with < 90% confidence
	DETECTION_MIN_CONFIDENCE = 0.9


############################################################
#  Dataset
############################################################

class MahjongTileDataset(utils.Dataset):

	def load_mahjongTile(self, dataset_dir, subset):
		"""Load a subset of the MahjongTile dataset.
		dataset_dir: Root directory of the dataset.
		subset: Subset to load: train or val
		"""
		# Add classes. We have only one class to add.
		self.add_class("mahjongTile", 1, "mahjongTile")

		# Train or validation dataset?
		assert subset in ["train", "val"]

		coco_json = json.load(open(os.path.join(dataset_dir, subset + ".json")))
		dataset_dir = os.path.join(dataset_dir, subset)

		# Add the class names using the base method from utils.Dataset
		source_name = "coco_like_dataset"
		for category in coco_json['categories']:
			class_id = category['id']
			class_name = category['name']
			if class_id < 1:
				print('Error: Class id for "{}" cannot be less than one. (0 is reserved for the background)'.format(
					class_name))
				return

		# Get all annotations
		annotations = {}
		for annotation in coco_json['annotations']:
			image_id = annotation['image_id']
			if image_id not in annotations:
				annotations[image_id] = []
			annotations[image_id].append(annotation)

		# Get all images and add them to the dataset
		seen_images = {}
		for image in coco_json['images']:
			image_id = image['id']
			if image_id in seen_images:
				print("Warning: Skipping duplicate image id: {}".format(image))
			else:
				seen_images[image_id] = image
				try:
					image_file_name = image['file_name']
					image_width = image['width']
					image_height = image['height']
					image_path = os.path.abspath(os.path.join(dataset_dir, image_file_name))
					image_annotations = annotations[image_id]

					# Add the image using the base method from utils.Dataset
					self.add_image(
						source="mahjongTile",
						image_id=image_id,
						path=image_path,
						width=image_width,
						height=image_height,
						annotations=image_annotations
					)
				except KeyError as key:
					print("Warning: Skipping image (id: {}) with missing key: {}".format(image_id, key))

	def load_mask(self, image_id):
		""" Load instance masks for the given image.
		MaskRCNN expects masks in the form of a bitmap [height, width, instances].
		Args:
			image_id: The id of the image to load masks for
		Returns:
			masks: A bool array of shape [height, width, instance count] with
				one mask per instance.
			class_ids: a 1D array of class IDs of the instance masks.
		"""
		image_info = self.image_info[image_id]
		annotations = image_info['annotations']
		instance_masks: list[np.ndarray] = []

		for annotation in annotations:
			class_id = annotation['category_id']
			RLE = annotation['segmentation']

			if type(RLE) == dict:
				# Polygon
				# Convert RLE to binary mask
				mask = mask_util.decode(RLE)
			else:
				mask = Image.new('1', (image_info['width'], image_info['height']))
				mask_draw = ImageDraw.ImageDraw(mask, '1')
				mask_draw.polygon(RLE[0], fill=1)
			
			bool_array = np.array(mask) > 0
			instance_masks.append(bool_array)

		mask = np.dstack(instance_masks)

		return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

	def image_reference(self, image_id):
		"""Return the path of the image."""
		info = self.image_info[image_id]
		if info["source"] == "mahjongTile":
			return info["path"]
		else:
			super(self.__class__, self).image_reference(image_id)


def train(model):
	"""Train the model."""
	# Training dataset.
	dataset_train = MahjongTileDataset()
	dataset_train.load_mahjongTile(args.dataset, "train")
	dataset_train.prepare()

	# Validation dataset
	dataset_val = MahjongTileDataset()
	dataset_val.load_mahjongTile(args.dataset, "val")
	dataset_val.prepare()

	# *** This training schedule is an example. Update to your needs ***
	# Since we're using a very small dataset, and starting from
	# COCO trained weights, we don't need to train too long. Also,
	# no need to train all layers, just the heads should do it.
	print("Training network heads")
	model.train(dataset_train, dataset_val,
				learning_rate=config.LEARNING_RATE,
				epochs=5,
				layers='heads')


def color_splash(image, mask):
	"""Apply color splash effect.
	image: RGB image [height, width, 3]
	mask: instance segmentation mask [height, width, instance count]

	Returns result image.
	"""
	# Make a grayscale copy of the image. The grayscale copy still
	# has 3 RGB channels, though.
	gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
	# Copy color pixels from the original color image where mask is set
	if mask.shape[-1] > 0:
		# We're treating all instances as one, so collapse the mask into one layer
		mask = (np.sum(mask, -1, keepdims=True) >= 1)
		splash = np.where(mask, image, gray).astype(np.uint8)
	else:
		splash = gray.astype(np.uint8)
	return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
	assert image_path or video_path

	# Image or video?
	if image_path:
		# Run model detection and generate the color splash effect
		print("Running on {}".format(args.image))
		# Read image
		image = skimage.io.imread(args.image)
		# Detect objects
		r = model.detect([image], verbose=1)[0]
		# Color splash
		splash = color_splash(image, r['masks'])
		# Save output
		file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
		skimage.io.imsave(file_name, splash)
	elif video_path:
		import cv2
		# Video capture
		vcapture = cv2.VideoCapture(video_path)
		width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
		height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
		fps = vcapture.get(cv2.CAP_PROP_FPS)

		# Define codec and create video writer
		file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
		vwriter = cv2.VideoWriter(file_name,
								  cv2.VideoWriter_fourcc(*'MJPG'),
								  fps, (width, height))

		count = 0
		success = True
		while success:
			print("frame: ", count)
			# Read next image
			success, image = vcapture.read()
			if success:
				# OpenCV returns images as BGR, convert to RGB
				image = image[..., ::-1]
				# Detect objects
				r = model.detect([image], verbose=0)[0]
				# Color splash
				splash = color_splash(image, r['masks'])
				# RGB -> BGR to save image to video
				splash = splash[..., ::-1]
				# Add image to video writer
				vwriter.write(splash)
				count += 1
		vwriter.release()
	print("Saved to ", file_name)


############################################################
#  Training
############################################################

if __name__ == '__main__':
	import argparse

	# Parse command line arguments
	parser = argparse.ArgumentParser(
		description='Train Mask R-CNN to detect mahjongTiles.')
	parser.add_argument("command",
						metavar="<command>",
						help="'train' or 'splash'")
	parser.add_argument('--dataset', required=False,
						metavar="/path/to/mahjongTile/dataset/",
						help='Directory of the MahjongTile dataset')
	parser.add_argument('--weights', required=True,
						metavar="/path/to/weights.h5",
						help="Path to weights .h5 file or 'coco'")
	parser.add_argument('--logs', required=False,
						default=DEFAULT_LOGS_DIR,
						metavar="/path/to/logs/",
						help='Logs and checkpoints directory (default=logs/)')
	parser.add_argument('--image', required=False,
						metavar="path or URL to image",
						help='Image to apply the color splash effect on')
	parser.add_argument('--video', required=False,
						metavar="path or URL to video",
						help='Video to apply the color splash effect on')
	args = parser.parse_args()

	# Validate arguments
	if args.command == "train":
		assert args.dataset, "Argument --dataset is required for training"
	elif args.command == "splash":
		assert args.image or args.video,\
			   "Provide --image or --video to apply color splash"

	print("Weights: ", args.weights)
	print("Dataset: ", args.dataset)
	print("Logs: ", args.logs)

	# Configurations
	if args.command == "train":
		config = MahjongTileConfig()
	else:
		class InferenceConfig(MahjongTileConfig):
			# Set batch size to 1 since we'll be running inference on
			# one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
			GPU_COUNT = 1
			IMAGES_PER_GPU = 1
		config = InferenceConfig()
	config.display()

	# Create model
	if args.command == "train":
		model = modellib.MaskRCNN(mode="training", config=config,
								  model_dir=args.logs)
	else:
		model = modellib.MaskRCNN(mode="inference", config=config,
								  model_dir=args.logs)

	# Select weights file to load
	if args.weights.lower() == "coco":
		weights_path = COCO_WEIGHTS_PATH
		# Download weights file
		if not os.path.exists(weights_path):
			utils.download_trained_weights(weights_path)
	elif args.weights.lower() == "last":
		# Find last trained weights
		weights_path = model.find_last()
	elif args.weights.lower() == "imagenet":
		# Start from ImageNet trained weights
		weights_path = model.get_imagenet_weights()
	else:
		weights_path = args.weights

	# Load weights
	print("Loading weights ", weights_path)
	if args.weights.lower() == "coco":
		# Exclude the last layers because they require a matching
		# number of classes
		model.load_weights(weights_path, by_name=True, exclude=[
			"mrcnn_class_logits", "mrcnn_bbox_fc",
			"mrcnn_bbox", "mrcnn_mask"])
	else:
		model.load_weights(weights_path, by_name=True)

	# Train or evaluate
	if args.command == "train":
		train(model)
	elif args.command == "splash":
		detect_and_color_splash(model, image_path=args.image,
								video_path=args.video)
	else:
		print("'{}' is not recognized. "
			  "Use 'train' or 'splash'".format(args.command))
