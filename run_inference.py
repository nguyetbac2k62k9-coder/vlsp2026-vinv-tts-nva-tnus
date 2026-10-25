#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VLSP 2026 ViNV-TTS Submission Engine
Team: nva_tnus
Model: OmniVoice (discrete masked diffusion, bidirectional Transformer) + LoRA checkpoint-2500
Notes: (1) No fixed random seed is set; minor audio variation between runs is expected.
       (2) Native OmniVoice BPE tokenizer is used without custom token expansion;
           LoRA adapts representations directly on competition tag subwords.
"""

import os
import sys
import csv
import logging
import traceback
import numpy as np
import soundfile as sf
import torch

# --- TƯƠNG THÍCH TRANSFORMERS VỚI PYTORCH 2.5.1 (CHỐNG LỖI torch.accelerator) ---
if not hasattr(torch, "accelerator"):
    class _Accelerator:
        @staticmethod
        def current_accelerator():
            return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        @staticmethod
        def is_available():
            return torch.cuda.is_available()
        @staticmethod
        def device_count():
            return torch.cuda.device_count() if torch.cuda.is_available() else 0
        @staticmethod
        def current_device_index():
            return torch.cuda.current_device() if torch.cuda.is_available() else 0
        @staticmethod
        def synchronize():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    torch.accelerator = _Accelerator
# -------------------------------------------------------------------------------

import torchaudio
import torchaudio.transforms as T
from peft import PeftModel
from omnivoice import OmniVoice

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nva_tnus")

INPUT_DIR = os.environ.get("INPUT_DIR", "/input")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "/output")
INPUT_CSV = os.path.join(INPUT_DIR, "input.csv")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "output.csv")
OUTPUT_AUDIO_DIR = os.path.join(OUTPUT_DIR, "audios")

MODEL_BASE_DIR = os.environ.get(
    "MODEL_BASE_DIR", 
    "/app/models/omnivoice_base" if os.path.exists("/app/models/omnivoice_base") else "models/omnivoice_base"
)
LORA_DIR = os.environ.get(
    "LORA_DIR", 
    "/app/models/checkpoint-2500" if os.path.exists("/app/models/checkpoint-2500") else "models/checkpoint-2500"
)

def load_lora_adapter(model, adapter_path, merge=True):
    """Nạp và hợp nhất LoRA adapter bằng PEFT trực tiếp"""
    logger.info(f"--> Đang gắn LoRA Adapter từ {adapter_path}...")
    peft_model = PeftModel.from_pretrained(model, adapter_path)
    if merge:
        return peft_model.merge_and_unload()
    return peft_model

def preprocess_ref_audio(audio_path, target_sr=24000):
    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    if sr != target_sr:
        resampler = T.Resample(
            orig_freq=sr,
            new_freq=target_sr,
            resampling_method="sinc_interp_kaiser",
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            beta=14.769656459379492
        )
        waveform = resampler(waveform)
    max_val = torch.max(torch.abs(waveform))
    if max_val > 1e-4:
        waveform = waveform / max_val * 0.95
    temp_path = f"/tmp/ref_{os.path.basename(audio_path)}"
    sf.write(temp_path, waveform.squeeze(0).cpu().numpy(), target_sr, subtype="PCM_16")
    return temp_path

def get_adaptive_params(text):
    t = text.lower()
    if "[throatclearing]" in t:
        return 3.8
    elif "[laughter]" in t:
        return 3.5
    elif "[sniff]" in t:
        return 2.8
    elif "[breathing]" in t:
        return 2.0
    return 2.0

def main():
    logger.info(">>> [nva_tnus] KHỞI ĐỘNG HỆ THỐNG SUY LUẬN CONTAINER...")
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    logger.info(f"--> Thiết bị phần cứng: {device}")
    os.makedirs(OUTPUT_AUDIO_DIR, exist_ok=True)

    try:
        logger.info(f"--> Đang nạp mô hình OmniVoice Base offline từ {MODEL_BASE_DIR}...")
        model = OmniVoice.from_pretrained(
            MODEL_BASE_DIR,
            device_map=device,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True
        )
        if os.path.exists(LORA_DIR):
            model = load_lora_adapter(model, LORA_DIR)
        logger.info("--> [nva_tnus] MÔ HÌNH ĐÃ SẴN SÀNG!")
    except Exception as e:
        logger.error(f"Lỗi khi nạp model: {e}")
        traceback.print_exc()
        sys.exit(1)

    if not os.path.exists(INPUT_CSV):
        logger.error(f"Không tìm thấy file: {INPUT_CSV}")
        sys.exit(1)

    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        fieldnames = list(reader[0].keys()) if reader else []

    if "output_audio_path" not in fieldnames:
        fieldnames.append("output_audio_path")

    output_rows = []
    total = len(reader)
    logger.info(f"--> Bắt đầu sinh âm thanh cho {total} mẫu câu...")

    for idx, row in enumerate(reader, 1):
        item_id = str(row.get("id", "")).strip()
        ref_text = row.get("ref_text", "")
        ref_audio_rel = row.get("ref_audio", "")
        target_text = row.get("target_text", "")
        out_filename = f"{item_id}.wav"
        out_rel_path = f"audios/{out_filename}"
        out_full_path = os.path.join(OUTPUT_AUDIO_DIR, out_filename)

        logger.info(f"[{idx}/{total}] Đang xử lý ID: {item_id} | Text: {target_text[:35]}...")

        success = False
        try:
            full_ref_audio = os.path.join(INPUT_DIR, ref_audio_rel)
            clean_ref = preprocess_ref_audio(full_ref_audio)
            guidance_scale = get_adaptive_params(target_text)

            with torch.no_grad():
                audio_out = model.generate(
                    text=target_text,
                    language="vi",
                    ref_audio=clean_ref,
                    ref_text=ref_text,
                    guidance_scale=guidance_scale,
                    num_step=28
                )
                wav_data = audio_out[0]
                if torch.is_tensor(wav_data):
                    wav_data = wav_data.cpu().numpy()

            peak = np.max(np.abs(wav_data))
            if peak > 1e-4:
                wav_data = (wav_data / peak) * 0.95

            sf.write(out_full_path, wav_data, 24000, subtype="PCM_16")
            success = True
        except Exception as e:
            logger.warning(f"--> [BỎ QUA MẪU LỖI {item_id}]: {e}")

        new_row = dict(row)
        new_row["output_audio_path"] = out_rel_path if success else ""
        output_rows.append(new_row)

    logger.info("--> Đang xuất file output.csv...")
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    logger.info(">>> [nva_tnus] HOÀN TẤT TOÀN BỘ QUÁ TRÌNH THÀNH CÔNG RỰC RỠ!")
    sys.exit(0)

if __name__ == "__main__":
    main()
