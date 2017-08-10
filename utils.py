import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy
import cv2

class BoundBox:
    def __init__(self, class_num):
        self.x, self.y, self.w, self.h, self.c = 0., 0., 0., 0., 0.
        self.probs = np.zeros((class_num,))
        
    def iou(self, box):
        intersection = self.intersect(box)
        union = self.w*self.h + box.w*box.h - intersection
        return intersection/union
        
    def intersect(self, box):
        width  = self.__overlap([self.x-self.w/2, self.x+self.w/2], [box.x-box.w/2, box.x+box.w/2])
        height = self.__overlap([self.y-self.h/2, self.y+self.h/2], [box.y-box.h/2, box.y+box.h/2])
        return width * height
        
    def __overlap(self, interval_a, interval_b):
        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2,x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2,x4) - x3

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def interpret_netout(image, netout):
    boxes = []

    # interpret the output by the network
    for row in range(GRID_H):
        for col in range(GRID_W):
            for b in range(BOX):
                box = BoundBox(CLASS)

                # first 5 weights for x, y, w, h and confidence
                box.x, box.y, box.w, box.h, box.c = netout[row,col,b,:5]

                box.x = (col + sigmoid(box.x)) / GRID_W
                box.y = (row + sigmoid(box.y)) / GRID_H
                box.w = ANCHORS[2 * b + 0] * np.exp(box.w) / GRID_W
                box.h = ANCHORS[2 * b + 1] * np.exp(box.h) / GRID_H
                box.c = sigmoid(box.c)

                # last 20 weights for class likelihoods
                classes = netout[row,col,b,5:]
                box.probs = softmax(classes) * box.c
                box.probs *= box.probs > THRESHOLD

                boxes.append(box)

    # suppress non-maximal boxes
    for c in range(CLASS):
        sorted_indices = list(reversed(np.argsort([box.probs[c] for box in boxes])))

        for i in xrange(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].probs[c] == 0: 
                continue
            else:
                for j in xrange(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if boxes[index_i].iou(boxes[index_j]) >= 0.4:
                        boxes[index_j].probs[c] = 0

    # draw the boxes using a threshold
    for box in boxes:
        max_indx = np.argmax(box.probs)
        max_prob = box.probs[max_indx]
        
        if max_prob > THRESHOLD:
            xmin  = int((box.x - box.w/2) * image.shape[1])
            xmax  = int((box.x + box.w/2) * image.shape[1])
            ymin  = int((box.y - box.h/2) * image.shape[0])
            ymax  = int((box.y + box.h/2) * image.shape[0])


            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), COLORS[max_indx], 2)
            cv2.putText(image, LABELS[max_indx], (xmin, ymin - 12), 0, 1e-3 * image.shape[0], (0,255,0), 2)
            
    return image

def parse_annotation(ann_dir):
    all_img = []
    
    for ann in os.listdir(ann_dir):
        img = {'object':[]}
        
        tree = ET.parse(ann_dir + ann)
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                all_img += [img]
                img['filename'] = elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        
                        if obj['name'] in LABELS:
                            img['object'] += [obj]
                        else:
                            break
                            
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
                        
    return all_img

def aug_img(train_instance):
    path = train_instance['filename']
    all_obj = copy.deepcopy(train_instance['object'][:])
    img = cv2.imread(img_dir + path)
    h, w, c = img.shape

    # scale the image
    scale = np.random.uniform() / 10. + 1.
    img = cv2.resize(img, (0,0), fx = scale, fy = scale)

    # translate the image
    max_offx = (scale-1.) * w
    max_offy = (scale-1.) * h
    offx = int(np.random.uniform() * max_offx)
    offy = int(np.random.uniform() * max_offy)
    img = img[offy : (offy + h), offx : (offx + w)]

    # flip the image
    flip = np.random.binomial(1, .5)
    if flip > 0.5: img = cv2.flip(img, 1)

    # re-color
    t  = [np.random.uniform()]
    t += [np.random.uniform()]
    t += [np.random.uniform()]
    t = np.array(t)

    img = img * (1 + t)
    img = img / (255. * 2.)

    # resize the image to standard size
    img = cv2.resize(img, (NORM_H, NORM_W))
    img = img[:,:,::-1]
    
    # fix object's position and size
    for obj in all_obj:
        for attr in ['xmin', 'xmax']:
            obj[attr] = int(obj[attr] * scale - offx)
            obj[attr] = int(obj[attr] * float(NORM_W) / w)
            obj[attr] = max(min(obj[attr], NORM_W), 0)
            
        for attr in ['ymin', 'ymax']:
            obj[attr] = int(obj[attr] * scale - offy)
            obj[attr] = int(obj[attr] * float(NORM_H) / h)
            obj[attr] = max(min(obj[attr], NORM_H), 0)
            
        if flip > 0.5:
            xmin = obj['xmin']
            obj['xmin'] = NORM_W - obj['xmax']
            obj['xmax'] = NORM_W - xmin
    
    return img, all_obj

def data_gen(all_img, batch_size):
    num_img = len(all_img)
    shuffled_indices = np.random.permutation(np.arange(num_img))
    l_bound = 0
    r_bound = batch_size if batch_size < num_img else num_img
    
    while True:
        if l_bound == r_bound:
            l_bound  = 0
            r_bound = batch_size if batch_size < num_img else num_img
            shuffled_indices = np.random.permutation(np.arange(num_img))
        
        batch_size = r_bound - l_bound
        currt_inst = 0
        x_batch = np.zeros((batch_size, NORM_W, NORM_H, 3))
        y_batch = np.zeros((batch_size, GRID_W, GRID_H, BOX, 5+CLASS))
        
        for index in shuffled_indices[l_bound:r_bound]:
            train_instance = all_img[index]
            
            # augment input image and fix object's position and size
            img, all_obj = aug_img(train_instance)
            #for obj in all_obj:
            #    cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']), (obj['xmax'],obj['ymax']), (1,1,0), 3)
            #plt.imshow(img); plt.show()
            
            # construct output from object's position and size
            for obj in all_obj:    
                box = []
                center_x = .5*(obj['xmin'] + obj['xmax']) #xmin, xmax
                center_x = center_x / (float(NORM_W) / GRID_W)
                center_y = .5*(obj['ymin'] + obj['ymax']) #ymin, ymax
                center_y = center_y / (float(NORM_H) / GRID_H)
                
                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))
                
                if grid_x < GRID_W and grid_y < GRID_H:
                    obj_indx = LABELS.index(obj['name'])
                    box = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax']]
                    
                    y_batch[currt_inst, grid_y, grid_x, :, 0:4]        = BOX * [box]
                    y_batch[currt_inst, grid_y, grid_x, :, 4  ]        = BOX * [1.]
                    y_batch[currt_inst, grid_y, grid_x, :, 5: ]        = BOX * [[0.]*CLASS]
                    y_batch[currt_inst, grid_y, grid_x, :, 5+obj_indx] = 1.0
                
            # concatenate batch input from the image
            x_batch[currt_inst] = img
            currt_inst += 1
            
            del img, all_obj
        
        yield x_batch, y_batch
        
        l_bound  = r_bound
        r_bound = r_bound + batch_size
        if r_bound > num_img: r_bound = num_img

def sigmoid(x):
    return 1. / (1.  + np.exp(-x))

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)