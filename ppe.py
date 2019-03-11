# -*- coding: utf-8 -*-
"""
Created on Sat Feb 09 10:59:46 2019

@author: Ian
"""

# USAGE
# python lettuce.py --mode train
# python lettuce.py --mode investigate
# python lettuce.py --mode predict --image examples/.jpg
# python lettuce.py --mode predict --image examples/.jpg  --weights logs/....

#IMPORT NECESSARY PACKAGES
#########################################################################
#create additional training data by applying random transformations
#reduces overfitting and allows greater generalizability
from imgaug import augmenters as iaa

#subclassing the Config class to derive configuration for training
from mrcnn.config import Config

#contains the mask-rcnn model itself
from mrcnn import model as modellib

#visualize output predictions of the mask-rcnn
from mrcnn import visualize

#various utilities leveraged
from mrcnn import utils
from imutils import paths
import numpy as np 
import argparse
import imutils
import skimage
import random
import json
import cv2
import os

#utilities for capturing video streams
from imutils.video import VideoStream
from imutils.video import FPS
import datetime
import time

# initialize the dataset path, images path, and annotations file path
DATASET_PATH = os.path.abspath("ppe")
IMAGES_PATH = os.path.sep.join([DATASET_PATH, "images"])
ANNOT_PATH = os.path.sep.join([DATASET_PATH, "via_region_data.json"])

# initialize the amount of data to use for training
TRAINING_SPLIT = 0.75

# grab all image paths, then randomly select indexes for both training
# and validation
IMAGE_PATHS = sorted(list(paths.list_images(IMAGES_PATH)))
idxs = list(range(0, len(IMAGE_PATHS)))
random.seed(42)
random.shuffle(idxs)
i = int(len(idxs) * TRAINING_SPLIT)
trainIdxs = idxs[:i]
valIdxs = idxs[i:]

# initialize the class names dictionary
CLASS_NAMES = {1: "Helmet", 2: "No Helmet", 3: "Vest", 4: "No Vest"}

# initialize the path to the Mask R-CNN pre-trained on COCO
COCO_PATH = "mask_rcnn_coco.h5"

# initialize the name of the directory where logs and output model
# snapshots will be stored
LOGS_AND_MODEL_DIR = os.path.sep.join([DATASET_PATH, "logs_ppe"])

class PPEConfig(Config):
	# give the configuration a recognizable name
	NAME = "PPE"

	# set the number of GPUs to use training along with the number of
	# images per GPU (which may have to be tuned depending on how
	# much memory your GPU has)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# set the number of steps per training epoch
	STEPS_PER_EPOCH = len(trainIdxs) // (IMAGES_PER_GPU * GPU_COUNT)

	# number of classes (+1 for the background)
	NUM_CLASSES = len(CLASS_NAMES) + 1

class PPEInferenceConfig(PPEConfig):
	# set the number of GPUs and images per GPU (which may be
	# different values than the ones used for training)
	GPU_COUNT = 1
	IMAGES_PER_GPU = 1

	# set the minimum detection confidence (used to prune out false
	# positive detections)
	DETECTION_MIN_CONFIDENCE = 0.9

class PPEDataset(utils.Dataset):
    def __init__(self, imagePaths, annotPath, classNames, width=1024):
		# call the parent constructor
        super().__init__(self)

		# store the image paths and class names along with the width
		# we'll resize images to
        self.imagePaths = imagePaths
        self.classNames = classNames
        self.width = width

		# load the annotation data
        self.annots = self.load_annotation_data(annotPath)

    def load_annotation_data(self, annotPath):
		# load the contents of the annotation JSON file (created
		# using the VIA tool) and initialize the annotations dictionary
        annotations = json.loads(open(annotPath).read())
        annots = {}

		# loop over the file ID and annotations themselves (values)
        for (fileID, data) in sorted(annotations.items()):
			# store the data in the dictionary using the filename as
			# the key
            #annots.append(data)
            annots[data["filename"]] = data

		# return the annotations dictionary
        return annots
    
    def load_ppe(self, idxs):
		# loop over all class names and add each to the Lettuce
		# dataset
        for (classID, label) in self.classNames.items():
            self.add_class("PPE", classID, label)
        
		# loop over the image path indexes
        for i in idxs:
		   	# extract the image filename to serve as the unique
			# image ID
            imagePath = self.imagePaths[i]
            filename = imagePath.split(os.path.sep)[-1]

			# load the image and resize it so we can determine its
			# width and height (unfortunately VIA does not embed
			# this information directly in the annotation file
            image = cv2.imread(imagePath)
            (origH, origW) = image.shape[:2]
            image = imutils.resize(image, width=self.width)
            (newH, newW) = image.shape[:2]
            
            #print(self.annots)
            
            polygons, names = [], []
            
            #print(self.annots)
            
            for r in self.annots[str(i)+".jpg"]["regions"]:
                polygons.append(r["shape_attributes"])            
                names.append(r["region_attributes"])  
                
                
            # add the image to the dataset
            self.add_image("PPE",
                           image_id=filename, 
                           height=newH,
                           width=newW,
                           orig_width=origW, 
                           orig_height=origH, 
                           polygons=polygons, 
                           names=names, 
                           path=imagePath)      

    
    def load_image(self, imageID):
        """ grab the image path, load it, and convert from BGR to RGB colour channel ordering
        """
        p = self.image_info[imageID]["path"]
        image = cv2.imread(p)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # resize the image, preserving the aspect ratio
        image = imutils.resize(image, width=self.width)
        return image

    def load_mask(self, imageID):
        """ Generates instance masks for an image.
        
        Returns:
        masks: A bool array of shape [height, width, num_instances] with one mask per instance
        classIDs: a 1D array of class IDs of the instance masks 
        

    { 
         "0.jpg4422650":
             {
                  "filename":"0.jpg", 
                  "size":4422650,
                  "regions":
                      [{
                          "shape_attributes":
                              {
                                  "name": "polygon",
                                  "all_points_x": [...], 
                                  "all_points_y": [...],
                              }, 
                          "region_attributes":
                              {
                                  "Lettuce": "1"
                              }
                      }],
                  "file_attributes": {}
             }    
    }    
    
        """
        #grab the image info and then grab the annotation data for
        #the current image based on the unique ID 
        info = self.image_info[imageID]
        class_names = info["names"]
        #annot = self.annots[info["id"]]
        #print(info["polygons"])
        
        #print(info["names"])
        
        #print(len(info["polygons"]))
        
        if info["source"]!= "PPE":
            return super(self.__class__, self).load_mask(imageID)

		# allocate memory for our [height, width, num_instances] array
		# where each "instance" effectively has its own "channel"
        mask = np.zeros((info["height"], info["width"], len(info["polygons"])), dtype="uint8")
        
        all_newY, all_newX = [], []
        
        #there are n instance dictionaries in polygons (list of dictionaries)
        #each instance dictionary has 3 keys
        #"name", "all_points_x", "all_points_y"
        for i, p in enumerate(info["polygons"]):
			# allocate memory for the region mask
            #regionMask = np.zeros(masks.shape[:2], dtype="uint8")

            ratio = info["width"] / float(info["orig_width"])
			    
            for pix in p["all_points_y"]:
                all_newY.append(int(pix * ratio))
            
            for pix in p["all_points_x"]:
                all_newX.append(int(pix * ratio))
                
            #get the indexes of pixels inside the polygon and set them to 1
            rr,cc = skimage.draw.polygon(all_newY, all_newX, shape=mask.shape)
            #rr,cc = skimage.draw.polygon(p["all_points_y"], p["all_points_x"])
            mask[rr, cc, i] = 1
            
        class_ids = np.zeros(len(info["polygons"]))

		# loop over each of the annotated region
        #there are n instance dictionaries in names (list of dictionaries)
        for i, p in enumerate(class_names):
            if p["Helmet"] == "1":
                class_ids[i] = 1
            if p["No Helmet"] == "2":
                class_ids[i] = 2
            if p["Vest"] == "3":
                class_ids[i] = 3
            if p["No Vest"] == "4":
                class_ids[i] = 4
                
        class_ids = class_ids.astype(int)

		# return the mask array and class IDs
        return mask.astype(np.bool), class_ids
			
############################################################################################################
if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--mode", required=True, help="either 'train', 'predict', or 'investigate'")
    ap.add_argument("-w", "--weights", help="optional path to pretrained weights")
    ap.add_argument("-i", "--input", help="optional path to input video file")
    args = vars(ap.parse_args())

	# check to see if we are training the Mask R-CNN
    if args["mode"] == "train":
		# load the training dataset
        trainDataset = PPEDataset(IMAGE_PATHS, ANNOT_PATH, CLASS_NAMES)
        trainDataset.load_ppe(trainIdxs)
        trainDataset.prepare()

		# load the validation dataset
        valDataset = PPEDataset(IMAGE_PATHS, ANNOT_PATH, CLASS_NAMES)
        valDataset.load_ppe(valIdxs)
        valDataset.prepare()

		# initialize the training configuration
        config = PPEConfig()
        config.display()

		# initialize the model and load the COCO weights so we can perform fine-tuning
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=LOGS_AND_MODEL_DIR)
        model.load_weights(COCO_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])

		# train *just* the layer heads
        model.train(trainDataset, valDataset, epochs=10, layers="heads", learning_rate=config.LEARNING_RATE)

		# unfreeze the body of the network and train *all* layers
        model.train(trainDataset, valDataset, epochs=20, layers="all", learning_rate=config.LEARNING_RATE / 10)

	# check to see if we are predicting using a trained Mask R-CNN
    elif args["mode"] == "predict":
		# initialize the inference configuration
        config = PPEInferenceConfig()

		# initialize the Mask R-CNN model for inference
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=LOGS_AND_MODEL_DIR)

		# load our trained Mask R-CNN
        weights = args["weights"] if args["weights"] \
        else model.find_last()
        model.load_weights(weights, by_name=True)

		#load input image
        if not args.get("input", False):
            print("starting video stream...")
            cap = VideoStream(src=0).start()
            time.sleep(2.0)
        else:
			#otherwise, grab a reference to the video file
            print("opening video file")
            cap = cv2.VideoCapture(args["input"])
            
        fps = FPS().start()
        
        class_names = ["BG", "Helmet", "No Helmet", "Vest", "No Vest"]

		#classes to keep the same mask in frames, generate colors for masks
        colors = visualize.random_colors(len(class_names))

		#llop over frames from the video stream
        while True:
			
			#read frame
            image = cap.read()
            image = image[1] if args.get("input", False) else image

			#if we are viewing a video and we did not grab a frame 
            #then we have reached the end of the video 
            if args["input"] is not None and image is None:
                break

			#resize the image and convert it from BGR to RGB channel ordering
            image = imutils.resize(image, width=1024)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
			#image = image[..., ::-1]
		
			# perform a forward pass of the network to obtain the results
            r = model.detect([image], verbose=1)[0]
            """
            
            # loop over of the detected object's bounding boxes and masks, drawing each as we go along
            for i in range(0, r["rois"].shape[0]):
                mask = r["masks"][:, :, i]
                image = visualize.apply_mask(image, mask, (1.0, 0.0, 0.0), alpha=0.5)
                image = visualize.draw_box(image, r["rois"][i], (1.0, 0.0, 0.0))
                
            # convert the image back to BGR so we can use OpenCV's drawing functions
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # loop over the predicted scores and class labels
            for i in range(0, len(r["scores"])):
				# extract the bounding box information, class ID, label,
				# and predicted probability from the results
                (startY, startX, endY, end) = r["rois"][i]
                classID = r["class_ids"][i]
                label = CLASS_NAMES[classID]
                score = r["scores"][i]
                
                # draw the class label and score on the image
                text = "{}: {:.4f} at {}".format(label, score, str(time.strftime("%H:%M:%S")))
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv2.putText(image, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            """
            
			# resize the image so it more easily fits on our screen
            image = imutils.resize(image, width=512)
            
            output = visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'],
										class_names, r['scores'], colors=colors, real_time=True)

			# show the output image
            cv2.imshow("Output", output)
			#cv2.imshow("Output", image)

			#interrupt trigger by pressing q to interrupt the OpenCV program
            ch = cv2.waitKey(1)
            if ch & 0xFF == ord("q"):
                break
            
            fps.update()
            fps.stop()

		#Cleanup
        cv2.waitKey(0)
        
        if not args.get("input", False):
            cap.stop()
            
        else:
            cap.release()
            
        cv2.destroyAllWindows()
		
	# check to see if we are investigating our images and mask
    elif args["mode"] == "investigate":
		# load the training dataset
        trainDataset = PPEDataset(IMAGE_PATHS, ANNOT_PATH, CLASS_NAMES)
        trainDataset.load_ppe(trainIdxs)
        trainDataset.prepare()

		# load the 0-th training image and corresponding masks and
		# class IDs in the masks
        image = trainDataset.load_image(0)
        (masks, classIDs) = trainDataset.load_mask(0)

		# show the image spatial dimensions which is HxWxC
        print("image shape: {}".format(image.shape))

		# show the masks shape which should have the same width and
		# height of the images but the third dimension should be
		# equal to the total number of instances in the image itself
        print("masks shape: {}".format(masks.shape))

		# show the length of the class IDs list along with the values
		# inside the list -- the length of the list should be equal
		# to the number of instances dimension in the 'masks' array
        print("class IDs length: {}".format(len(classIDs)))
        print("class IDs: {}".format(classIDs))
        print("\n")

		# determine a sample of training image indexes and loop over them
        for i in np.random.choice(trainDataset.image_ids, 3):
			# load the image and masks for the sampled image
            print("investigating image index: {}".format(i))
            image = trainDataset.load_image(i)
            (masks, classIDs) = trainDataset.load_mask(i)
            print("image shape: {}".format(image.shape))
            print("masks shape: {}".format(masks.shape))
            print("class IDs length: {}".format(len(classIDs)))
            print("class IDs: {}".format(classIDs))
            print("\n")
            
			# visualize the masks for the current image
            visualize.display_top_masks(image, masks, classIDs, trainDataset.class_names)
