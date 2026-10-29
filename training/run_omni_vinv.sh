#!/bin/bash
# ==============================================================================
# Pipeline Huấn luyện OmniVoice với Dữ liệu vinv-tts (Non-Verbal Speech)
# ==============================================================================

set -euo pipefail

stage=0
stop_stage=1

# Cấu hình GPU (Ví dụ: "0" cho 1 GPU, "0,1" cho 2 GPUs)
GPU_IDS="0"
NUM_GPUS=1

# Đường dẫn file dữ liệu JSONL đã tạo bởi prepare_data.py
TRAIN_JSONL="data/omnivoice/train.jsonl"
DEV_JSONL="data/omnivoice/dev.jsonl"

# Thư mục lưu audio tokens đã trích xuất
TOKEN_DIR="data/omnivoice/tokens"

# Pretrained Audio Tokenizer (Higgs Audio v2)
TOKENIZER_PATH="eustlb/higgs-audio-v2-tokenizer"

# File cấu hình training và data
TRAIN_CONFIG="configs/train_vinv_lora.json"
DATA_CONFIG="configs/data_vinv.json"

# Thư mục lưu checkpoint sau khi train
OUTPUT_DIR="exp/omnivoice_vinv_lora"

export PYTHONPATH="$(pwd)/OmniVoice:${PYTHONPATH:-}"

# Giai đoạn 0: Trích xuất Audio Codec Tokens thành WebDataset shards
if [ $stage -le 0 ] && [ $stop_stage -ge 0 ]; then
    echo "=========================================================="
    echo " Stage 0: Tokenizing audio (Higgs Audio v2 Tokenizer)"
    echo "=========================================================="

    for split_jsonl_path in "${TRAIN_JSONL}" "${DEV_JSONL}"; do
        [ -z "${split_jsonl_path}" ] && continue
        if [ "${split_jsonl_path}" = "${TRAIN_JSONL}" ]; then
            split="train"
        else
            split="dev"
        fi

        echo "-> Đang trích xuất tokens cho tập [${split}] từ ${split_jsonl_path}..."
        mkdir -p "${TOKEN_DIR}/${split}"

        CUDA_VISIBLE_DEVICES=${GPU_IDS} \
            python -m omnivoice.scripts.extract_audio_tokens \
            --input_jsonl "${split_jsonl_path}" \
            --tar_output_pattern "${TOKEN_DIR}/${split}/audios/shard-%06d.tar" \
            --jsonl_output_pattern "${TOKEN_DIR}/${split}/txts/shard-%06d.jsonl" \
            --tokenizer_path "${TOKENIZER_PATH}" \
            --nj_per_gpu 2 \
            --shuffle True

        echo "-> Hoàn thành tập [${split}]. Manifest: ${TOKEN_DIR}/${split}/data.lst"
    done
fi

# Giai đoạn 1: Huấn luyện OmniVoice với LoRA
if [ $stage -le 1 ] && [ $stop_stage -ge 1 ]; then
    echo "=========================================================="
    echo " Stage 1: Fine-tuning OmniVoice với LoRA Adapter"
    echo "=========================================================="

    accelerate launch \
        --gpu_ids "${GPU_IDS}" \
        --num_processes ${NUM_GPUS} \
        -m omnivoice.cli.train \
        --train_config "${TRAIN_CONFIG}" \
        --data_config "${DATA_CONFIG}" \
        --output_dir "${OUTPUT_DIR}"

    echo "-> Huấn luyện hoàn tất! Checkpoints được lưu tại: ${OUTPUT_DIR}"
fi
