#! /usr/bin/env python

import argparse
import os
import numpy as np
from preprocessing import parse_annotation
from frontend_evaluate import YOLO
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
    '--inputVal',
    help='val image folder')

argparser.add_argument(
    '-a',
    '--annotVal',
    help='val annotation folder')    

argparser.add_argument(
    '-j',
    '--inputTest',
    help='test image folder')

argparser.add_argument(
    '-t',
    '--annotTest',
    help='test annotation folder')    

def _main_(args):
    config_path = args.conf

    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())

    ###############################
    #   Parse the annotations 
    ###############################

    valid_imgs, valid_labels = parse_annotation(args.annotVal, 
                                                args.inputVal, 
                                                config['model']['labels'])

    test_imgs, test_labels = parse_annotation(args.annotTest, 
                                              args.inputTest, 
                                              config['model']['labels'])


    if len(config['model']['labels']) > 0:
        overlap_labels = set(config['model']['labels']).intersection(set(valid_labels.keys()))

        print('Seen labels:\t', valid_labels)
        print('Given labels:\t', config['model']['labels'])
        print('Overlap labels:\t', overlap_labels)           

        if len(overlap_labels) < len(config['model']['labels']):
            print('Some labels have no annotations! Please revise the list of labels in the config.json file!')
            return
    else:
        print('No labels are provided. Train on all seen labels.')
        config['model']['labels'] = valid_labels.keys()
        
    ###############################
    #   Construct the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load the pretrained weights (if any) 
    ###############################    

    print("Loading pre-trained weights in", args.weights)
    yolo.load_weights(args.weights)

    ###############################
    #   Start the training process 
    ###############################

    yolo.eval(valid_imgs         = valid_imgs,
               test_imgs          = test_imgs,               
               learning_rate      = config['train']['learning_rate'], 
               batch_size         = config['train']['batch_size'],
               object_scale       = config['train']['object_scale'],
               no_object_scale    = config['train']['no_object_scale'],
               coord_scale        = config['train']['coord_scale'],
               class_scale        = config['train']['class_scale'],
               debug              = config['train']['debug'])

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
