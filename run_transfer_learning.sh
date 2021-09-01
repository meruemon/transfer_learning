#!/bin/bash

# 訓練画像が保存されたフォルダ
IMAGES_DIR="/data/e-kikai/data/samples"
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
