from os import listdir
from os.path import isfile, join
import argparse
import cv2
import numpy as np
import sys
import os
import shutil
import random
import math
import argparse
import os
import numpy as np

from preprocessing import parse_annotation
from frontend import YOLO
import json

from utils import BoundBox, bbox_iou, get_feature_extractor

argparser = argparse.ArgumentParser()

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')

width_in_cfg_file = 416.
height_in_cfg_file = 416.
grid_h, grid_w = (13, 13)

def IOU(x,centroids):
    similarities = []
    k = len(centroids)
    for centroid in centroids:
        c_w,c_h = centroid
        w,h = x
        if c_w>=w and c_h>=h:
            similarity = w*h/(c_w*c_h)
        elif c_w>=w and c_h<=h:
            similarity = w*c_h/(w*h + (c_w-w)*c_h)
        elif c_w<=w and c_h>=h:
            similarity = c_w*h/(w*h + c_w*(c_h-h))
        else: #means both w,h are bigger than c_w and c_h respectively
            similarity = (c_w*c_h)/(w*h)
        similarities.append(similarity) # will become (k,) shape
    return np.array(similarities)

def avg_IOU(X,centroids):
    n,d = X.shape
    sum = 0.
    for i in range(X.shape[0]):
        #note IOU() will return array which contains IoU for each centroid and X[i] // slightly ineffective, but I am too lazy
        sum+= max(IOU(X[i],centroids))
    return sum/n

def write_anchors_to_file(centroids,X,anchor_file):
    f = open(anchor_file,'w')

    anchors = centroids.copy()

    for i in range(anchors.shape[0]):
        anchors[i][0]*=width_in_cfg_file/32./grid_w
        anchors[i][1]*=height_in_cfg_file/32./grid_h

    widths = anchors[:,0]
    sorted_indices = np.argsort(widths)

    r = "anchors: [ "
    for i in sorted_indices[:-1]:
        r += '%0.10f,%0.10f,' % (anchors[i,0]/width_in_cfg_file,anchors[i,1]/width_in_cfg_file)

    #there should not be comma after last anchor, that's why
    r += '%0.10f,%0.10f' % (anchors[sorted_indices[-1:],0]/width_in_cfg_file/1.,anchors[sorted_indices[-1:],1]/width_in_cfg_file/1.)
    r += "]"

    print r
    print

def kmeans(X,centroids,eps,anchor_file):

    N = X.shape[0]
    iterations = 0
    k,dim = centroids.shape
    prev_assignments = np.ones(N)*(-1)
    iter = 0
    old_D = np.zeros((N,k))

    while True:
        D = []
        iter+=1
        for i in range(N):
            d = 1 - IOU(X[i],centroids)
            D.append(d)
        D = np.array(D) # D.shape = (N,k)

        print "iter {}: dists = {}".format(iter,np.sum(np.abs(old_D-D)))

        #assign samples to centroids
        assignments = np.argmin(D,axis=1)

        if (assignments == prev_assignments).all() :
            print "Centroids = ",centroids
            write_anchors_to_file(centroids,X,anchor_file)
            return

        #calculate new centroids
        centroid_sums=np.zeros((k,dim),np.float)
        for i in range(N):
            centroid_sums[assignments[i]]+=X[i]
        for j in range(k):
            centroids[j] = centroid_sums[j]/(np.sum(assignments==j))

        prev_assignments = assignments.copy()
        old_D = D.copy()

def main(argv):
    config_path = args.conf

    with open(config_path) as config_buffer:
        config = json.loads(config_buffer.read())

    grid_h, grid_w = get_feature_extractor(config['model']['architecture'], config['model']['input_size']).get_output_shape()
    width_in_cfg_file = int(config['model']['input_size'])
    height_in_cfg_file = int(config['model']['input_size'])

    train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                config['train']['train_image_folder'],
                                                config['model']['labels'])
    annotation_dims = []
    for image in train_imgs:
        i = image['object'][0]
        w = (float(i['xmax']) - float(i['xmin']))
        h = (float(i["ymax"]) - float(i['ymin']))
        annotation_dims.append(map(float,(w,h)))
    annotation_dims = np.array(annotation_dims)

    eps = 0.00005

    indices = [ random.randrange(annotation_dims.shape[0]) for i in range(5)]
    centroids = annotation_dims[indices]
    kmeans(annotation_dims,centroids,eps,"f1")

if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
