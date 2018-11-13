# code for Research


## Shot_detection
 CNN＋LSTMにてショットの自動分割を行うPG

### LSTM.py
LSTMのモデルと学習を行う


### load.py
CSVのファイルを読み込むクラス


### all_extra_featue.py / extra_feature.py
学習済みCNNから特徴ベクトルを抽出するPG
モデルと動画を設定する必要がある


### extractor.py
学習済みCNNをロードしてくるクラス
ここに新しいCNNモデルなんかを今後追加して書いてくといいかもね


### marge.py
各動画から抽出したCSVファイルを一つに統合する



## Description


## Requirement

* Ubuntu 16.04LTS
* python 3.5.2
* numpy 1.14.4
* tensorflow-gpu 1.8.0
* tensorboard 1.8.0
* keras 2.2.0






