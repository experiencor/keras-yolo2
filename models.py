from keras.models import Model
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
import tensorflow as tf
import numpy as np
import cv2
from keras.layers.merge import concatenate
from keras.optimizers import SGD, Adam, RMSprop
from preprocessing import BatchGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from utils import BoundBox

FULL_YOLO_FEATURE_PATH = "full_yolo_features.h5" # should be hosted on a server
TINY_YOLO_FEATURE_PATH = "tiny_yolo_features.h5" # should be hosted on a server

class YOLO(object):
    def __init__(self, architecture,
                       input_size, 
                       labels, 
                       max_box_per_image,
                       anchors):

        self.input_size = input_size
        self.grid_h = input_size/32
        self.grid_w = input_size/32
        
        self.anchors  = anchors
        self.labels   = list(labels)
        self.nb_class = len(self.labels)
        self.nb_box   = 5
        self.class_wt = np.ones(self.nb_class, dtype='float32')

        self.no_object_scale = 5.
        self.object_scale    = 1.
        self.coord_scale     = 1.
        self.class_scale     = 1.

        self.max_box_per_image = max_box_per_image

        ##########################
        # Make the model
        ##########################

        if architecture == 'Full Yolo':
            # the function to implement the orgnization layer (thanks to github.com/allanzelener/YAD2K)
            def space_to_depth_x2(x):
                return tf.space_to_depth(x, block_size=2)

            input_image = Input(shape=(self.input_size, self.input_size, 3))
            self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))            

            # Layer 1
            x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
            x = BatchNormalization(name='norm_1')(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

            # Layer 2
            x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
            x = BatchNormalization(name='norm_2')(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

            # Layer 3
            x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
            x = BatchNormalization(name='norm_3')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 4
            x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
            x = BatchNormalization(name='norm_4')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 5
            x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
            x = BatchNormalization(name='norm_5')(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

            # Layer 6
            x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
            x = BatchNormalization(name='norm_6')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 7
            x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
            x = BatchNormalization(name='norm_7')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 8
            x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False, input_shape=(416,416,3))(x)
            x = BatchNormalization(name='norm_8')(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

            # Layer 9
            x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
            x = BatchNormalization(name='norm_9')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 10
            x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
            x = BatchNormalization(name='norm_10')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 11
            x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
            x = BatchNormalization(name='norm_11')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 12
            x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
            x = BatchNormalization(name='norm_12')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 13
            x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
            x = BatchNormalization(name='norm_13')(x)
            x = LeakyReLU(alpha=0.1)(x)

            skip_connection = x

            x = MaxPooling2D(pool_size=(2, 2))(x)

            # Layer 14
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
            x = BatchNormalization(name='norm_14')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 15
            x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
            x = BatchNormalization(name='norm_15')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 16
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
            x = BatchNormalization(name='norm_16')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 17
            x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
            x = BatchNormalization(name='norm_17')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 18
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
            x = BatchNormalization(name='norm_18')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 19
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
            x = BatchNormalization(name='norm_19')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 20
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
            x = BatchNormalization(name='norm_20')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 21
            skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
            skip_connection = BatchNormalization(name='norm_21')(skip_connection)
            skip_connection = LeakyReLU(alpha=0.1)(skip_connection)
            skip_connection = Lambda(space_to_depth_x2)(skip_connection)

            x = concatenate([skip_connection, x])

            # Layer 22
            x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
            x = BatchNormalization(name='norm_22')(x)
            x = LeakyReLU(alpha=0.1)(x)

            # Layer 23
            x = Conv2D(self.nb_box * (4 + 1 + self.nb_class), (1,1), strides=(1,1), padding='same', name='conv_23', kernel_initializer='he_normal')(x)
            output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(x)

            # a small hack to allow true_boxes to be registered when Keras build the model 
            # for more information: https://github.com/fchollet/keras/issues/2790
            output = Lambda(lambda args: args[0])([output, self.true_boxes])

            self.model = Model([input_image, self.true_boxes], output)

            # load the pretrained weights of all layers except the last convolutional layer
            self.model.load_weights(FULL_YOLO_FEATURE_PATH, by_name=True)

        elif architecture == 'Tiny Yolo':
            # Layer 1
            input_image = Input(shape=(self.input_size, self.input_size, 3))
            self.true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))            

            # Layer 1
            x = Conv2D(16, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
            x = BatchNormalization(name='norm_1')(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

            # Layer 2 - 5
            for i in range(0,4):
                x = Conv2D(32*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
                x = BatchNormalization(name='norm_' + str(i+2))(x)
                x = LeakyReLU(alpha=0.1)(x)
                x = MaxPooling2D(pool_size=(2, 2))(x)

            # Layer 6
            x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
            x = BatchNormalization(name='norm_6')(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

            # Layer 7 - 8
            for i in range(0,2):
                x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
                x = BatchNormalization(name='norm_' + str(i+7))(x)
                x = LeakyReLU(alpha=0.1)(x)

            # Layer 9
            x = Conv2D(self.nb_box * (4 + 1 + self.nb_class), (1,1), strides=(1,1), padding='same', name='conv_9', kernel_initializer='he_normal')(x)
            output = Reshape((self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))(x)

            output = Lambda(lambda args: args[0])([output, self.true_boxes])

            self.model = Model([input_image, self.true_boxes], output)

            # load the pretrained weights of all layers except the last convolutional layer
            self.model.load_weights(TINY_YOLO_FEATURE_PATH, by_name=True)     
                  
        else:
            raise Exception('Architecture not supported! Only support Full Yolo and Tiny Yolo at the moment!')

        self.model.summary()

    def custom_loss(self, y_true, y_pred):
        mask_shape = tf.shape(y_true)[:4]
        
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, 5, 1])
        
        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
        
        seen = tf.Variable(0.)
        
        total_ap = tf.Variable(0.)
        
        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1,1,1,self.nb_box,2])
        
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        
        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]
        
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell
        
        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
        
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half       
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        
        true_box_conf = iou_scores * y_true[..., 4]
        
        ### adjust class probabilities
        true_box_class = tf.to_int32(y_true[..., 5])
        
        """
        Determine the masks
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale
        
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale
        
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale
        
        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale       
        
        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale/2.)
        seen = tf.assign_add(seen, 1.)
        
        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_bs), 
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                       true_box_wh + tf.ones_like(true_box_wh) * np.reshape(self.anchors, [1,1,1,self.nb_box,2]) * no_boxes_mask, 
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       coord_mask])
        
        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
        
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
        
        loss = loss_xy + loss_wh + loss_conf + loss_class
        
        nb_true_box = tf.reduce_sum(y_true[..., 4])
        nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))
        
        total_ap = tf.assign_add(total_ap, nb_pred_box/nb_true_box) 
        
        #loss = tf.Print(loss, [loss_xy, loss_wh, loss_conf, loss_class, loss, total_ap/seen], message='DEBUG', summarize=1000)
        
        return loss

    def load_weights(self, weight_path):
        self.model.load_weights(weight_path)

    def preprocess(self, image):
        input_image = cv2.resize(image, (self.input_size, self.input_size))
        input_image = input_image / 255.
        input_image = input_image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)

        return input_image

    def predict(self, image):
        input_image = self.preprocess(image)
        dummy_array = dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.model.predict([input_image, dummy_array])[0]
        boxes  = self.decode_netout(netout)
        
        return boxes

    def bbox_iou(self, box1, box2):
        x1_min  = box1.x - box1.w/2
        x1_max  = box1.x + box1.w/2
        y1_min  = box1.y - box1.h/2
        y1_max  = box1.y + box1.h/2
        
        x2_min  = box2.x - box2.w/2
        x2_max  = box2.x + box2.w/2
        y2_min  = box2.y - box2.h/2
        y2_max  = box2.y + box2.h/2
        
        intersect_w = self.interval_overlap([x1_min, x1_max], [x2_min, x2_max])
        intersect_h = self.interval_overlap([y1_min, y1_max], [y2_min, y2_max])
        
        intersect = intersect_w * intersect_h
        
        union = box1.w * box1.h + box2.w * box2.h - intersect
        
        return float(intersect) / union
        
    def interval_overlap(self, interval_a, interval_b):
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

    def decode_netout(self, netout, obj_threshold=0.3, nms_threshold=0.3):
        grid_h, grid_w, nb_box = netout.shape[:3]

        boxes = []
        
        # decode the output by the network
        netout[..., 4]  = self.sigmoid(netout[..., 4])
        netout[..., 5:] = netout[..., 4][..., np.newaxis] * self.softmax(netout[..., 5:])
        netout[..., 5:] *= netout[..., 5:] > obj_threshold
        
        for row in range(grid_h):
            for col in range(grid_w):
                for b in range(nb_box):
                    # from 4th element onwards are confidence and class classes
                    classes = netout[row,col,b,5:]
                    
                    if np.sum(classes) > 0:
                        # first 4 elements are x, y, w, and h
                        x, y, w, h = netout[row,col,b,:4]

                        x = (col + self.sigmoid(x)) / grid_w # center position, unit: image width
                        y = (row + self.sigmoid(y)) / grid_h # center position, unit: image height
                        w = self.anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                        h = self.anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                        confidence = netout[row,col,b,4]
                        
                        box = BoundBox(x, y, w, h, confidence, classes)
                        
                        boxes.append(box)

        # suppress non-maximal boxes
        for c in range(self.nb_class):
            sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

            for i in xrange(len(sorted_indices)):
                index_i = sorted_indices[i]
                
                if boxes[index_i].classes[c] == 0: 
                    continue
                else:
                    for j in xrange(i+1, len(sorted_indices)):
                        index_j = sorted_indices[j]
                        
                        if self.bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                            boxes[index_j].classes[c] = 0
                            
        # remove the boxes which are less likely than a obj_threshold
        boxes = [box for box in boxes if box.get_score() > obj_threshold]
        
        return boxes

    def sigmoid(self, x):
        return 1. / (1. + np.exp(-x))

    def softmax(self, x, axis=-1, t=-100.):
        x = x - np.max(x)
        
        if np.min(x) < t:
            x = x/np.min(x)*t
            
        e_x = np.exp(x)
        
        return e_x / e_x.sum(axis, keepdims=True)

    def train(self, train_imgs, valid_imgs, nb_epoch, learning_rate, batch_size, warmup_bs):
        self.batch_size = batch_size
        self.warmup_bs  = warmup_bs

        generator_config = {
            'IMAGE_H'         : self.input_size, 
            'IMAGE_W'         : self.input_size,
            'GRID_H'          : self.grid_h,  
            'GRID_W'          : self.grid_w,
            'BOX'             : self.nb_box,
            'LABELS'          : self.labels,
            'CLASS'           : len(self.labels),
            'ANCHORS'         : self.anchors,
            'BATCH_SIZE'      : self.batch_size,
            'TRUE_BOX_BUFFER' : self.max_box_per_image,
        }            

        ############################################
        # Compile the model
        ############################################

        optimizer = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        self.model.compile(loss=self.custom_loss, optimizer=optimizer)

        ############################################
        # Make train and validation generators
        ############################################

        train_batch = BatchGenerator(train_imgs, generator_config)
        valid_batch = BatchGenerator(valid_imgs, generator_config, jitter=False)

        ############################################
        # Make a few callbacks
        ############################################

        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=0.001, 
                           patience=3, 
                           mode='min', 
                           verbose=1)
        checkpoint = ModelCheckpoint('best_weights.h5', 
                                     monitor='val_loss', 
                                     verbose=1, 
                                     save_best_only=True, 
                                     mode='min', 
                                     period=1)
        tensorboard = TensorBoard(log_dir='~/logs/coco', 
                                  histogram_freq=0, 
                                  write_graph=True, 
                                  write_images=False)

        ############################################
        # Start the training process
        ############################################        

        self.model.fit_generator(generator        = train_batch.get_generator(), 
                                 steps_per_epoch  = train_batch.get_dateset_size(), 
                                 epochs           = nb_epoch, 
                                 verbose          = 1,
                                 validation_data  = valid_batch.get_generator(),
                                 validation_steps = valid_batch.get_dateset_size(),
                                 callbacks        = [early_stop, checkpoint, tensorboard], 
                                 max_queue_size   = 3)
