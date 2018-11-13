# Shot_detection
 CNN＋LSTMにてショットの自動分割を行う

### LSTM.py
LSTMのモデルと学習を行う

### 3Dconv.py
3D Convolutional Neural Networks

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
- Software
    - python3.6.3
    - tensorflow==1.7.0
    - keras==2.1.5
    - numpy==1.14.0
    - matplotlib==2.2.2
    - opencv-python==3.4.1.15
