# 類似画像検索システム

## 環境構築

プログラムをGithubからダウンロードする．

```bash
$ cd ~
$ mkdir Programs
$ cd Programs
$ git clone https://github.com/meruemon/transfer_learning.git
$ cd transfer_learning
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

## 特徴量抽出

画像から画像特徴量と呼ばれる特徴量ベクトルを抽出し，機械学習手法を用いて画像認識を実現する手法が主流となった．
機械学習は，クラスラベルを付与した大量の学習サンプルを必要とするが，ルールベースの手法のように研究者がいくつかのルールを設計する
必要がないため，汎用性の高い画像認識を実現できる．2000年代になると，画像特徴量としてScale-Invariant Feature Transform (SIFT)
やHistogram of Oriented Gradients (HOG) のように研究者の知見に基づいて設計した特徴量が盛んに研究されていた． 
このように設計された特徴量は**handcrafted feature**と呼ばれる．そして，2010年代では学習により特徴抽出過程を自動獲得する
深層学習 (Deep learning) か登場した．handcrafted featureは，研究者の知見に基づいて設計したアルゴリズムにより特徴量を
抽出・表現していたため最適であるとは限らない．
深層学習は，認識に有効な特徴量の抽出処理を自動化することができる新しいアプローチである．


深層学習で用いられるニューラルネットワークは分類器だけではなく，特徴抽出器としての有用性が知られている．
具体的に， ネットワークの中間層から出力されたベクトルを特徴量とみなして，**Deep feature**と呼ばれる．


事前学習済み深層ニューラルネットワークInception-V3を使用してDeep featureを抽出する手順を述べる．


### 実行

`run_incv3_features_extraction.sh`が画像特徴量を抽出するシェルスクリプトである．
次のコマンドを入力するとプログラムが実行される．

```bash
$ sh run_incv3_features_extraction.sh
```

中身は以下の通りである．実行されるpythonコードは`process_images.py`と`process_images_v2`の2つである．
４行目から15行目の変数`IMAGES_DIR`，`MODEL_DIR`，`FINETUNED_MODEL_DIR`，`FINETUNED_MODEL_NAME`，`OUTPUT_DIR`，`FINETUNED_OUTPUT_DIR`は
パラメータである．特に，`IMAGES_DIR`に保存されている画像を対象に，画像特徴量を抽出するため，ここに指定したフォルダの中に
画像が存在しないと特徴量は抽出されない．`FINETUNED_MODEL_DIR`には，ファインチューニング後に生成された重みパラメータファイル`**.pb`が
保存されているフォルダを指定する．抽出された特徴量は，画像ファイル名を含むファイル名で`OUPUT_DIT`と`FINETUNED_OUTPUT_DIR`に
保存される．

```shell
#!/bin/bash

# 特徴抽出対象の画像が保存されたフォルダ（自分の環境に合わせて変更）
IMAGES_DIR="/home/yoshida/Programs/e-kikai/transfer_learning/CBIR/utils/static/img"
# ImageNetで学習されたパラメータを保存するフォルダ（変更する必要なし）
MODEL_DIR="./saved_models/inception_v3"
# ファインチューニングしたパラメータを保存したフォルダ（自分の環境に合わせて変更）
FINETUNED_MODEL_DIR="./saved_models/20210828"
# ファインチューニングしたパラメータのファイル名
# FINETUNED_MODEL_DIRで指定したフォルダの中に保存されているファイル名と一緒にする
FINETUNED_MODEL_NAME="freezed_model_50.pb"
# 画像特徴ベクトルを保存するフォルダ（自分の環境に合わせて変更）
OUTPUT_DIR="./static/vectors"
# ファインチューンされた画像特徴ベクトルを保存するフォルダ（自分の環境に合わせて変更）
FINETUNED_OUTPUT_DIR="./static/finetuned_vectors"

# saved_modelsの中に重みパラメータを保存するフォルダを作成
mkdir -p ${MODEL_DIR}
# 画像特徴ベクトルを保存するフォルダを作成
mkdir -p ${OUTPUT_DIR}
mkdir -p ${FINETUNED_OUTPUT_DIR}

# 画像特徴抽出を行う（ファインチューニング無）
python process_images.py --model_dir "${MODEL_DIR}" --image_files "${IMAGES_DIR}" --output_folder "${OUTPUT_DIR}"

# 画像特徴抽出を行う（ファインチューニング有）
python process_images_v2.py --model_dir "${FINETUNED_MODEL_DIR}" --image_files "${IMAGES_DIR}"\
                            --output_folder "${FINETUNED_OUTPUT_DIR}" --model_name "${FINETUNED_MODEL_NAME}"
```

一度の実行で，`process_images.py`と`process_images_v2.py`の両方が実行されるので，不要であれば
どちらか一方の該当行をコメントアウトする．