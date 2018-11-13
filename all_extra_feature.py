# coding: utf-8

"""
どうすか

ラベリングファイルを読み込んで，
image_path, binary_label, extra feature vector, 
というCSVファイルを作成する．

"""

import numpy as np
import csv
import os, sys
import glob
from extractor import Extractor
#バックエンドをインポート
from keras.backend import tensorflow_backend as backend


model_name = 'fc2'

LABELING_DIR = '/media/futami/HDD1/DATASET_KINGDOM/Scene/shot_list/'
NEW_LABELING_DIR = './feature_list/'+ str(model_name) +'_feature_list/'

if not os.path.exists(NEW_LABELING_DIR):
    os.makedirs(NEW_LABELING_DIR)


path_list = glob.glob(LABELING_DIR + '*.csv')
print(path_list)

for path in path_list:
    read_data_list = []
    new_data_list = []
    #class_num = 0

    print(path)
    basename = os.path.basename(path)
    
    # ラベル付けしたCSVファイルを読み込む
    with open(path, 'r') as f:

        reader = csv.reader(f)

        # ヘッダーがあるなら読み飛ばし
        #header = next(reader)

        for row in reader:
            read_data_list.append(row)
        
        print ("all labeling CSV file read !!!\n")

    model = Extractor()
    # データを作っていくぞ
    for read in read_data_list:
        
        # バイナリ用と多クラス用のラベリングを作成
        binary_label = int(float(read[1]))
        #if binary_label == 1:
        #    class_num += 1

        img_path = str(read[0])
        # ディレクトリを修正
        img_path = img_path.replace('./data', '/media/futami/HDD1/DATASET_KINGDOM/Scene')

        print(img_path)
        
        # 特徴ベクトルをnumpy形式で保存
        # 使用する特徴抽出器を選択
        feature = model.extract(img_path, model_name)
        feature_shape = str(feature.shape)

        feature = feature.tolist()

        feature.insert(0, img_path)
        feature.insert(1, binary_label)
        #feature.insert(2, class_num)

        new_data_list.append(feature)

        print ('extra feature: ' + img_path)

    # save labeling as csv file
    with open(NEW_LABELING_DIR + basename , 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['image_file', 'label_binary', model_name + ': '+feature_shape])
        """for new_data in new_data_list:
            writer.writerow(new_data)"""
        writer.writerows(new_data_list)

        print ("write featue" + basename)
        
    #処理終了時に下記をコール
    backend.clear_session()

    
