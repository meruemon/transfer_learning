#!/bin/bash

# 特徴抽出対処うの画像が保存されたフォルダ
IMAGES_DIR="/home/yoshida/Programs/e-kikai/transfer_learning/CBIR/utils/static/img"
# ImageNetで学習されたパラメータを保存するフォルダ（変更する必要なし）
MODEL_DIR="./saved_models/inception_v3"
# ファインチューニングしたパラメータを保存したフォルダ
FINETUNED_MODEL_DIR="./saved_models/20210828"
# ファインチューニングしたパラメータのファイル名
FINETUNED_MODEL_NAME="freezed_model_50.pb"
# 画像特徴ベクトルを保存するフォルダ
OUTPUT_DIR="/home/yoshida/Programs/e-kikai/transfer_learning/CBIR/utils/static/vectors"
# ファインチューンされた画像特徴ベクトルを保存するフォルダ
FINETUNED_OUTPUT_DIR="/home/yoshida/Programs/e-kikai/transfer_learning/CBIR/utils/static/finetuned_vectors"

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
