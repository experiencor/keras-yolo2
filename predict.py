#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
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
    '--input',
    help='path to an image or an video (mp4 format)')


def _main_(args):
 
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input
    isdir = os.path.isdir(image_path)

    if isdir:
        print image_path+os.listdir(image_path)[0]
        first_image_path = image_path + os.listdir(image_path)[0]

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)
    
     # Read (first) image and check the image depth.
    img_first = cv2.imread(image_path)
    isgrey = np.all(img_first[:,:,0] == img_first[:,:,1]) and  np.all(img_first[:,:,0] == img_first[:,:,2])
    if isgrey:
        depth = 1
    else:
        depth = 3

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                input_depth	    = depth,
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

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]

        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()  
    else:
        if isdir:
            paths = os.listdir(image_path)
        else:
            paths = [image_path]
        for image_path in paths:
            if depth == 3:
                image = cv2.imread(image_path)
            if depth == 1:
                image = cv2.imread(image_path,0)
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            print len(boxes), 'boxes are found'

            cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
