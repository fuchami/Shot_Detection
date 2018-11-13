# coding: utf-8

"""
学習済みのCNNをモデルから特徴量を抽出するだけの機会となる存在です．頑張れ

複数の特徴量を用いた比較を行う
VGG fc1/ fc2 
Inception v3 
"""

from keras.applications.xception import Xception
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model
import numpy as np

class Extractor():

    def __init__(self):
        
        base_model = VGG16(weights='imagenet')
        
        self.model_fc1 = Model(inputs=base_model.input,
                            outputs=base_model.get_layer('fc1').output
        )

        self.model_fc2 = Model(inputs=base_model.input,
                            outputs=base_model.get_layer('fc2').output
        )

        v3base_model = InceptionV3(weights='imagenet')
        self.model_v3 = Model(inputs=v3base_model.input,
                            outputs=v3base_model.get_layer('avg_pool').output
        )

        # avg_pool (GlobalAveragePooling2 (None, 2048) block14_sepconv2_act[0][0] 
        xbase_model = Xception(weights='imagenet')
        self.model_x = Model(input=xbase_model.input,
                            outputs=xbase_model.get_layer('avg_pool').output
        )


    def extract(self, image_path, model_name):
        img = image.load_img(image_path, target_size=(224, 224))

        # convert image to numpy
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)


        #features = self.model.predict(x)

        if model_name == 'fc1':
            features = self.model_fc1.predict(x)
        elif model_name == 'fc2':
            features = self.model_fc2.predict(x)
        elif model_name == 'v3':
            features = self.model_v3.predict(x)
        elif model_name == 'xception':
            features = self.model_x.predict(x)
        else:
            print ("please select extra feature model  'fc1' or 'fc2' or 'v3' or 'xception' ")
            
        print (features.shape)
        return features[0]
    