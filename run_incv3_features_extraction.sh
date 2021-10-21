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
OUTPUT_DIR="./CBIR/utils/static/vectors"
# ファインチューンされた画像特徴ベクトルを保存するフォルダ（自分の環境に合わせて変更）
FINETUNED_OUTPUT_DIR="./CBIR/utils/static/finetuned_vectors"

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
