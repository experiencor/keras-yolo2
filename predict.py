#! /usr/bin/env python

"""
This script takes in a configuration file and produces the best model as well as the list of labels
that appear in the train dataset. The configuration file is a json file and looks like this:

{
    "train_image_folder": "/home/husky/data/pascal/VOCdevkit/VOC2007/JPEGImages/",
    "train_annot_folder": "/home/husky/data/pascal/VOCdevkit/VOC2007/Annotations/",
    "valid_image_folder": "/home/husky/data/pascal/VOCdevkit/VOC2007/JPEGImages/",
    "valid_annot_folder": "/home/husky/data/pascal/VOCdevkit/VOC2007/Annotations/",
    "pretrained_feature": "yolo_feature.h5",

    "input_size": 416,
    "anchors": [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
    "batch_size": 2,
    "learning_rate": 1e-4,
    "max_box_per_image": 20,
    "nb_epoch": 10
}

The first 5 parameters are compulsory. Their names are self-explanatory.

The rest of the parameters can be left to the defaults.
"""

import argparse
import os
import cv2
import numpy as np
from preprocessing import parse_annotation
from utils import draw_boxes
from models import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=""

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

argparser.add_argument(
    '-w',
    '--weights',
    help='path to pretrained weights')

argparser.add_argument(
    '-i',
    '--image',
    help='path to an image')

def _main_(args):
 
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.image

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(architecture=config['model']['architecture'],
                input_size=config['model']['input_size'], 
                labels=config['model']['labels'], 
                max_box_per_image=config['model']['max_box_per_image'],
                anchors=config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    image = cv2.imread(image_path)
    boxes = yolo.predict(image)
    image = draw_boxes(image, boxes, config['model']['labels'])

    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)