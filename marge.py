# coding: utf-8

"""
複数のcsvファイルを１つに統合する

"""

import numpy as np
import os,sys
import glob
import csv

if __name__ == '__main__':
    
    path_list = glob.glob('./feature_list/fc2_feature_list/*.csv')
    marged_list = []
    class_num = 0

print(path_list)

for path in path_list:
    switch = 0
    with open(path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            #binary：先頭を1にする
            if switch == 0:
                row[1] = str(1)

            marged_list.append(row)
            switch+=1


with open('./feature_list/fc2_feature_list/marge_list.csv', 'w') as f:
    writer = csv.writer(f)
    writer.writerow(["image_file", "label_binary", "label_class", "fc1_feature"])
    writer.writerows(marged_list)




