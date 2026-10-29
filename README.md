# NVA-TNUS System: VLSP 2026 ViNV-TTS Shared Task

**Team:** NVA_TNUS  
**Affiliation:** Department of Mathematics and Informatics, Thai Nguyen University of Sciences (TNUS)  
**Task:** Vietnamese Non-Verbal Speech Synthesis and Zero-Shot Voice Cloning (ViNV-TTS)  
**Tracks:** Track A (Seen Multi-Speaker) & Track B (Unseen Zero-Shot Voice Cloning)

---

## 1. System Overview

This reproduce package contains the complete inference engine, Docker configuration, and reproducibility instructions for the **NVA-TNUS** primary submission to the VLSP 2026 ViNV-TTS Shared Task (submitted on October 12, 2026).

The system adapts **OmniVoice** — a discrete masked diffusion TTS model with a bidirectional Transformer backbone (initialized from Qwen3-0.6B) — via **LoRA** ($r=16, \alpha=32$, 2,500 fine-tuning steps) on the official multi-speaker ViNV-TTS training corpus.

### Key Technical Specifications

| Component | Value |
|---|---|
| Base model | `k2-fsa/OmniVoice` (discrete masked diffusion, bidirectional Transformer) |
| Adapter | LoRA `checkpoint-2500` ($r=16, \alpha=32$, all linear attention & MLP modules) |
| Decoding steps | `num_step = 28` (configured for latency reduction and real-time RTF < 1.0) |
| Precision | `float16` (CUDA) |
| Reference preprocessing | Mono downmix → 24 kHz sinc resampling (Kaiser $\beta=14.77$, width=64) → peak norm 0.95 |
| Tag-adaptive CFG scale | `[throatclearing]`=3.8, `[laughter]`=3.5, `[sniff]`=2.8, `[breathing]`/default=2.0 |
| Output format | 24 kHz, 16-bit PCM WAV, peak-normalized to 0.95 |
| Random seed | **Not fixed** (minor stochastic variation across runs is expected due to discrete diffusion sampling) |

> **Note on Tokenizer:** The model utilizes the native OmniVoice BPE tokenizer without vocabulary expansion; the 4 competition non-verbal tags are processed directly as standard subword sequences, and LoRA adapts the model representations.

---

## 2. Directory Structure

```
├── Dockerfile                   # Docker container definition (pinned dependencies)
├── run_inference.py             # Main batch inference engine (supports Docker & native Conda)
├── requirements.txt             # Pinned Python dependencies
├── training/                    # Complete Training & Fine-Tuning Pipeline
│   ├── prepare_data.py          # Data manifest preparation script
│   ├── run_omni_vinv.sh         # End-to-end training pipeline (Tokenize -> LoRA train)
│   ├── run_omni_pipeline.py     # Cross-platform Python pipeline runner
│   ├── configs/
│   │   ├── train_vinv_lora.json # LoRA hyperparameters (r=16, alpha=32, 2,500 steps)
│   │   └── data_vinv.json       # Shard data manifest config
│   └── README.md                # Detailed training instructions
├── models/
│   ├── omnivoice_base/          # Pretrained OmniVoice base weights (~4.9 GB)
│   └── checkpoint-2500/         # LoRA adapter weights (~190 MB, 2,500 fine-tuning steps)
└── README.md                    # System documentation and reproducibility guide
```

---

## 3. Model Checkpoint Download

The model weights (~5.1 GB total) must be downloaded and placed into the `models/` folder before building the Docker image.

### Option A: Hugging Face Hub (Recommended)

```bash
pip install huggingface_hub
python -c "
from huggingface_hub import snapshot_download
# Base model 
snapshot_download('nva-tnus/omnivoice-vinv-tts-base', local_dir='models/omnivoice_base')
# LoRA adapter checkpoint-2500
snapshot_download('nva-tnus/omnivoice-vinv-tts-checkpoint-2500', local_dir='models/checkpoint-2500')
"
```

### Option B: Google Drive Archive

Download the archive / models from: https://drive.google.com/drive/folders/1BkOTbSgQeoydhAF_OqRQ6yeMJTlVhwse  
Extract into `models/` such that:
- `models/omnivoice_base/config.json`
- `models/checkpoint-2500/adapter_config.json`
are directly accessible.

---

## 4. Reproduce via Docker (Recommended)

```bash
# Step 1: Ensure models are in place (see Section 3)
ls -la models/omnivoice_base/
ls -la models/checkpoint-2500/

# Step 2: Build the Docker image
docker build -t nva_tnus_vlsp2026 .

# Step 3: Run evaluation container
docker run --gpus all --rm \
    -v /path/to/test_input:/input:ro \
    -v /path/to/test_output:/output:rw \
    nva_tnus_vlsp2026
```

The container automatically reads `/input/input.csv` (columns: `id`, `ref_audio`, `ref_text`, `target_text`) and synthesizes 24 kHz audio into `/output/audios/*.wav` along with `/output/output.csv`.

---

## 5. Reproduce in Native Conda / Python Environment

```bash
conda create -n nva_tnus python=3.11 -y
conda activate nva_tnus

# Install PyTorch with CUDA 12.4
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install verified pinned dependencies from competition run
pip install transformers==5.17.0 accelerate==1.15.0 peft==0.21.0 soundfile==0.14.0 scipy==1.17.1 numpy==2.4.6

# Install OmniVoice from verified submission commit
pip install git+https://github.com/k2-fsa/OmniVoice.git@08be0b4ccbac3e13e374e86fbfead4b4cac343e2

# Set I/O paths and execute inference
export INPUT_DIR=/path/to/test_input
export OUTPUT_DIR=/path/to/test_output
python run_inference.py
```

---

## 6. Training & Fine-Tuning Pipeline

To reproduce model training from the raw ViNV-TTS dataset to `checkpoint-2500` (matching Section 4.3 of our paper):

```bash
# 1. Prepare data manifests (generates data/omnivoice/train.jsonl and dev.jsonl)
python training/prepare_data.py

# 2. Run end-to-end training pipeline
# Stage 0: Tokenize audio with Higgs Audio v2 -> Stage 1: LoRA Fine-Tuning (2,500 steps)
bash training/run_omni_vinv.sh

# Or execute cross-platform via Python runner:
python training/run_omni_pipeline.py 0 1
```

All hyperparameters (LoRA rank $r=16, \alpha=32$, learning rate $1\times 10^{-4}$, warmup, cosine scheduler) are defined in `training/configs/train_vinv_lora.json`. For more details, see `training/README.md`.

---

## 7. Software & Hardware Environment

- **Hardware:** NVIDIA GeForce RTX 3060 (12 GB VRAM)
- **CUDA Runtime:** 12.4
- **Python:** 3.11.16
- **PyTorch:** 2.5.1
- **Transformers:** 5.17.0
- **PEFT:** 0.21.0

---

## 8. License & Ethical Disclaimer

- **Base Model License:** Pre-trained OmniVoice base checkpoint is distributed under the Creative Commons Attribution-NonCommercial (CC-BY-NC 4.0) license.
- **Usage:** This software and adapted model weights are provided strictly for academic research within the VLSP 2026 ViNV-TTS Shared Task. Commercial use, voice spoofing, and unauthorized voice cloning are strictly prohibited.

---

## 9. Authors and Contact

- **Nguyet Do Anh** (Lead Author)
- **Anh Duong Thi Ngoc** (Co-Author)
- **Chung Tran Quang** (Co-Author)
- **Phuong Pham Ngoc\*** (Corresponding Author)

**Affiliation:** Department of Mathematics and Informatics, Thai Nguyen University of Sciences (TNUS), Thai Nguyen, Vietnam.

---

## 10. Citation

If you find this work or codebase helpful in your research, please cite our technical report:

```bibtex
@inproceedings{nva_tnus_vlsp2026,
  title     = {{NVA-TNUS} System for {VLSP} 2026 {ViNV-TTS} Shared Task: Vietnamese Non-Verbal Speech Synthesis and Zero-Shot Voice Cloning via Fine-Tuned Diffusion Language Models},
  author    = {Do Anh, Nguyet and Duong Thi Ngoc, Anh and Tran Quang, Chung and Pham Ngoc, Phuong},
  booktitle = {Proceedings of the 11th International Conference on Vietnamese Language and Speech Processing (VLSP 2026)},
  year      = {2026}
}
```