FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install pinned Python dependencies
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    transformers==5.17.0 \
    accelerate==1.15.0 \
    peft==0.21.0 \
    soundfile==0.14.0 \
    scipy==1.17.1 \
    numpy==2.4.6

# Install OmniVoice from verified submission commit
RUN pip install --no-cache-dir \
    git+https://github.com/k2-fsa/OmniVoice.git@08be0b4ccbac3e13e374e86fbfead4b4cac343e2

# NOTE: The 'models/' directory must be present before building.
# Models are mapped/copied into /app/models:
#   models/omnivoice_base/    - pretrained OmniVoice base weights
#   models/checkpoint-2500/   - LoRA adapter weights (2500 fine-tuning steps)
# Download instructions: see README.md Section 3.
COPY models /app/models
COPY run_inference.py /app/run_inference.py

ENTRYPOINT ["python", "/app/run_inference.py"]
