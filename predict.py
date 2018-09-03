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
import xml.etree.ElementTree as ET

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
    '--input',
    help='path to an image or an video (mp4 format)')

argparser.add_argument(
    '-a',
    '--annotFile',
    help='annotation File')   
  

def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

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
                               cv2.VideoWriter_fourcc(*'mp4v'), 
                               30.0, 
                               (frame_h, frame_w),True) #(frame_w, frame_h) # Virando video

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            image = np.rot90(image,3)
            image = image.copy() # Fix Bug np.rot90
            
            boxes = yolo.predict(image)
            #image = draw_boxes(image, boxes, config['model']['labels'], 20, 3.5, -90)
            image = draw_boxes(image, boxes, config['model']['labels'], 2, 1.1, -30)

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()  
    else:
        image = cv2.imread(image_path)

        if (args.annotFile != None):
            boxes_ann = []
            tree = ET.parse(args.annotFile)
            for elem in tree.iter():
                if 'object' in elem.tag or 'part' in elem.tag:
                    obj = {}
                    
                    for attr in list(elem):
                        if 'name' in attr.tag:
                            obj['name'] = attr.text

                            boxes_ann.append(obj)
                                
                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(round(float(dim.text)))
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(round(float(dim.text)))
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(round(float(dim.text)))
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(round(float(dim.text)))
            
            for box in boxes_ann:
                cv2.rectangle(image, (box['xmin'],box['ymin']), (box['xmax'],box['ymax']), (255,0,0), 30)

        boxes = yolo.predict(image)
        image = draw_boxes(image, boxes, config['model']['labels'], 30, 4.5, 35)

        print(len(boxes), 'boxes are found')

        cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
