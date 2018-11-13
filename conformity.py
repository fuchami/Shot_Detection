 # coding: utf-8

"""
Get_shot用のラベリングをScene_labeling用のラベリングに変換させる


**************************************************************************
before 

img_filename	label	histgram	CosSim	hist_diff	CosSim_diff
img0.jpg	0	0	0	0	0
img1.jpg	0	0.9999611333	0.985054	0.9999611333	0.9850537777

**************************************************************************

afeter
./data/frame/----/0.jpg
./data/frame/----/1.jpg

"""

import numpy as np
import csv
import os, sys

OLD_LABELING_DIR = '/home/futami/Desktop/Scene_detection/data/list/old/'
NEW_LABELING_DIR = '/home/futami/Desktop/Scene_detection/data/list/'

read_data_list = []
data_list = []
new_data_list =[]
cnt = 0

# 旧ラベル付けしたCSVファイルを読み込む
with open(OLD_LABELING_DIR + str(sys.argv[1]) + '.csv', 'r') as f:

    reader = csv.reader(f)

    #ヘッダー読み飛ばし
    header = next(reader)

    for row in reader:
        read_data_list.append(row)

    
    print ("all old labeling CSV fiel read...")


#　変換させるよ
for read in read_data_list:
    data_list = []

    print(read[0])
    str(read[0]).replace('img', '')
    print (read[0])
    frame = './data/frame/'+str(sys.argv[1]) +'/'+ str(cnt) + '.jpg'

    data_list.append(frame)
    data_list.append(read[1])

    new_data_list.append(data_list)
    cnt+=1

    print("conformityyy..." + frame)


# CSVファイルに保存
with open(NEW_LABELING_DIR + str(sys.argv[1]) + '.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerows(new_data_list)

    print ('write csv file !')
