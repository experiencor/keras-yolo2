''' if your custom backend  file is in another folder, you can use sys.path.append to add the
folder wheres the backend module from this repo is located, also it works for another imports '''
#import sys
#sys.path.append("path/to/backend")
from backend import BaseFeatureExtractor
from keras.models import Model
import tensorflow as tf
from keras.layers import Reshape, Conv2D, Input, MaxPooling2D, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

class SuperTinyYoloFeature(BaseFeatureExtractor):
    """
    It is a example from TinyTolo reduced around 16x times your size, also this network has
    4 maxPoolings instead 5 as the original, with 4 maxpoolings this network will generate a different
    grid size
    """
    def __init__(self, input_size):
        input_image = Input(shape=(input_size, input_size, 3))

        # Layer 1
        x = Conv2D(1, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
        x = BatchNormalization(name='norm_1')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 2 - 5
        for i in range(0,3):
            x = Conv2D(2*(2**i), (3,3), strides=(1,1), padding='same', name='conv_' + str(i+2), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+2))(x)
            x = LeakyReLU(alpha=0.1)(x)
            x = MaxPooling2D(pool_size=(2, 2))(x)

        # Layer 6
        x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
        x = BatchNormalization(name='norm_6')(x)
        x = LeakyReLU(alpha=0.1)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(1,1), padding='same')(x)

        # Layer 7 - 8
        for i in range(0,2):
            x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_' + str(i+7), use_bias=False)(x)
            x = BatchNormalization(name='norm_' + str(i+7))(x)
            x = LeakyReLU(alpha=0.1)(x)

        self.feature_extractor = Model(input_image, x)  

    def normalize(self, image):
        return image / 255.