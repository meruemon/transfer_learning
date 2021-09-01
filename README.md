# 類似画像検索システム

## 環境構築

プログラムをGithubからダウンロードする．

```bash
$ cd ~
$ mkdir Programs
$ cd Programs
$ git clone
$ cd CBIR
```

ログインユーザのIDを確認し，`docker/Dockerfile`46行目の`UID`に表示された番号を設定する．

```bash
$ id student
uid=1001(student) gid=1001(student) groups=1001(student)

Dockerfile

(略)
ENV USER student
ENV UID 1001
(略)
```

ベースイメージをダウンロードする．

```bash
docker pull nvcr.io/nvidia/tensorflow:19.12-tf1-py3
```

`docker pull`でダウンロードが拒否された場合，[NGC](https://ngc.nvidia.com/signin)からAPI Keyを確認して`nvci.io`にログインする．

```bash
$ docker login nvcr.io

Username: $oauthtoken
Password: <Your Key>
```

ベースイメージを基に，研究用イメージを作成する．

```bash
$ cd docker
$ docker build -t ubuntu18.04:tf1.15-py3
```

`docker-compose`コマンドでコンテナを作成する．

```bash
$ docker-compose up -d
$ docker exec -it cbir bash
```

以降は，全てコンテナ上で実行する．

## ファインチューニング

### 訓練データの準備

本研究では，ImageNetで事前学習されたInception-v3をベースネットワークとして，画像特徴量（ベクトル）を抽出して，類似画像検索に利用する．
学習済みのモデルの層の重みを初期値として，ファインチューニングを用いて層の重みを微調整する．

準備として，ファインチューニングを行う際の訓練データとなる画像を指定の様式に従って整理する．
Tensorflow(Keras)では，あるファルダにサブフォルダを作成し，それらに画像を保存すると，
サブフォルダの名前をクラスとして認識し，サブフォルダ内の画像をクラスに対応する訓練データとして読み込むことができる．
フォルダの名前は，日本語以外であれば，アルファベットまたは数字どちらとしても良い．

```
Folder
|
|-- SubfolderA -- img1
|              |- img2
|             
|-- SubfolderB -- img3
| 
```


### 実行

訓練データの準備ができたら，ImageNetで学習済みのInception-V3をファインチューニングする．
[シェルスクリプト](https://qiita.com/zayarwinttun/items/0dae4cb66d8f4bd2a337)の中に，pythonコードの実行手順を記載してるため
次のコマンドを入力するとプログラムが実行される．

```bash
$ sh run_transfer_learning.sh
```

`run_transfer_learning.sh`の中身は以下の通りである．実行されるpythonコードは`transfer_learning.py`と
`keras_to_tensorflow.py`の２つである．
４行目から12行目の変数`IMAGES_DIR`，`OUTPUT_MODEL`，`BATCH_NUM`，`EPOCHS`はパラメータである．
`IMAGES_DIR`に保存されている画像を読み込んでネットワークを訓練するため，
ここで指定したフォルダの中に画像が存在しないとプログラムは実行されない．

実行が成功したら，`OUTPUT_MODEL`に入力したフォルダの中に，`***.pb`が作成される．
`OUTPUT_MODEL`は自動で作成される．下のスクリプトでは，実行時の日付を変数`NOW`に代入し，
`saved_models`の中に日付を名前とするフォルダを`mkdir`コマンドで作成している．

```shell
#!/bin/bash

# 訓練画像が保存されたフォルダ
IMAGES_DIR="/home/student/Programs/data/training"
# 現在時刻を取得
NOW=$(date +"%Y%d%m")
# 訓練済みのInception-v3のパラメータを保存するフォルダ
OUTPUT_MODEL="./saved_models/${NOW}"
# バッチサイズ
BATCH_NUM=128
# エポック数
EPOCHS=50

# saved_modelsの中に実行時の時刻のフォルダを作成
mkdir -p ${OUTPUT_MODEL}
# ログ保存用のフォルダを作成
mkdir -p ./logs

# ファインチューニングを実行
python transfer_learning.py --images_dir "${IMAGES_DIR}" --output_model "${OUTPUT_MODEL}/saved_model_${EPOCHS}.h5" \
                             --batch_size ${BATCH_NUM} --epochs ${EPOCHS}

# Kerasで作成したモデルパラメータ(.h5)をTensorflowで読み込めるように(.pb)へ変換する
python keras_to_tensorflow.py --input_model "${OUTPUT_MODEL}/saved_model_${EPOCHS}.h5" \
                              --output_model "${OUTPUT_MODEL}/freezed_model_${EPOCHS}.pb"
```

