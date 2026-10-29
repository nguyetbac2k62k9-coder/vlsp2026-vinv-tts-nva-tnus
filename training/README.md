# NVA-TNUS Training Pipeline: VLSP 2026 ViNV-TTS Shared Task

**Team:** NVA_TNUS  
**Affiliation:** Department of Mathematics and Informatics, Thai Nguyen University of Sciences (TNUS)  
**Task:** Vietnamese Non-Verbal Speech Synthesis and Zero-Shot Voice Cloning (ViNV-TTS)  
**Model:** OmniVoice + LoRA Adaptation (Checkpoint-2500)

---

## 1. Directory Structure

```
training_package/
├── prepare_data.py          # Converts official ViNV-TTS audio/annotations to OmniVoice manifests
├── run_omni_vinv.sh         # Complete end-to-end training pipeline (Stage 0: Tokenize, Stage 1: LoRA train)
├── run_omni_pipeline.py     # Cross-platform Python execution pipeline
├── configs/
│   ├── train_vinv_lora.json # Hyperparameters for LoRA adaptation (r=16, alpha=32, 2,500 steps)
│   └── data_vinv.json       # WebDataset token manifest configuration
└── README.md                # This documentation
```

---

## 2. Training Workflow Overview

The training process follows two stages matching Section 4.3 of our Technical Report:

1. **Stage 0: Audio Tokenization**
   - Extracts discrete acoustic tokens from raw 24 kHz audio using the pretrained **Higgs Audio v2 Tokenizer** (`eustlb/higgs-audio-v2-tokenizer`, 8 RVQ codebooks at 25 fps).
   - Generates sharded WebDataset `.tar` archives and text `.jsonl` files for train (1,739 utterances) and dev (316 utterances) sets.

2. **Stage 1: Parameter-Efficient LoRA Fine-Tuning**
   - Fine-tunes **OmniVoice** using LoRA ($r=16, \alpha=32$, dropout $p=0.05$) targeting all linear attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`).
   - Unfreezes token embeddings and audio classification heads.
   - Optimizer: AdamW, peak learning rate $1 \times 10^{-4}$, weight decay 0.01, linear warmup over 50 steps (2%), and cosine decay.
   - Batch size: 768 tokens with gradient accumulation of 12 steps in `bfloat16` precision on an NVIDIA GeForce RTX 3060 (12 GB).
   - Training budget: 2,500 steps, saving checkpoints every 500 steps (`checkpoint-500`, `checkpoint-1000`, ..., `checkpoint-2500`).

---

## 3. Step-by-Step Reproduction Instructions

### Step 1: Environment Setup

```bash
# Create Conda environment
conda create -n omnivoice_train python=3.11 -y
conda activate omnivoice_train

# Install PyTorch with CUDA support
pip install torch==2.5.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124

# Install training dependencies
pip install transformers==5.17.0 accelerate==1.15.0 peft==0.21.0 soundfile==0.14.0 scipy webdataset

# Install OmniVoice from the verified competition commit
pip install git+https://github.com/k2-fsa/OmniVoice.git@08be0b4ccbac3e13e374e86fbfead4b4cac343e2
```

### Step 2: Data Preparation

Place the official competition dataset in `vinv-tts/` (containing `train/` and `dev/` folders), then run:

```bash
python prepare_data.py
```

This generates `data/omnivoice/train.jsonl` (1,739 samples) and `data/omnivoice/dev.jsonl` (316 samples).

### Step 3: Execute Training Pipeline

**Option A: Using Bash (Recommended for Linux/GPU Servers):**
```bash
chmod +x run_omni_vinv.sh
bash run_omni_vinv.sh
```

**Option B: Using Python Pipeline Runner:**
```bash
# Run both Stage 0 (Tokenize) and Stage 1 (LoRA Fine-tuning)
python run_omni_pipeline.py 0 1
```

Checkpoints will be automatically saved in `exp/omnivoice_vinv_lora/`, and the final submission checkpoint is `checkpoint-2500/`.

---

## 4. Hardware Requirements

- **Tested Hardware:** NVIDIA GeForce RTX 3060 (12 GB VRAM).
- **RAM:** Minimum 16 GB system memory.
- **Disk Space:** ~20 GB for WebDataset audio token shards and checkpoints.

---

## 5. Contact

- **Lead Author:** Nguyet Do Anh — `doanhnguyet@tnus.edu.vn`
- **Corresponding Author:** Phuong Pham Ngoc — `phuongpn@tnus.edu.vn`
- **Affiliation:** Department of Mathematics and Informatics, Thai Nguyen University of Sciences (TNUS), Vietnam
