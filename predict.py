#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

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

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    print weights_path
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    image = cv2.imread(image_path)
    boxes = yolo.predict(image)
    image = draw_boxes(image, boxes, config['model']['labels'])

    print len(boxes), 'boxes are found'

    cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)