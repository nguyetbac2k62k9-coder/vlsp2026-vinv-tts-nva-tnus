#!/usr/bin/env python3
import os
import sys
import subprocess
from pathlib import Path

def run_command(cmd, env=None):
    print(f"\n[RUNNING] {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        print(f"[ERROR] Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)

def main():
    base_dir = Path(__file__).resolve().parent
    omni_repo = base_dir / "OmniVoice"
    
    # Setup PYTHONPATH
    current_env = os.environ.copy()
    pythonpath = str(omni_repo)
    if "PYTHONPATH" in current_env:
        pythonpath += os.pathsep + current_env["PYTHONPATH"]
    current_env["PYTHONPATH"] = pythonpath

    stage = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    stop_stage = int(sys.argv[2]) if len(sys.argv) > 2 else 1

    train_jsonl = base_dir / "data" / "omnivoice" / "train.jsonl"
    dev_jsonl = base_dir / "data" / "omnivoice" / "dev.jsonl"
    token_dir = base_dir / "data" / "omnivoice" / "tokens"
    tokenizer_path = "eustlb/higgs-audio-v2-tokenizer"
    train_config = base_dir / "configs" / "train_vinv_lora.json"
    data_config = base_dir / "configs" / "data_vinv.json"
    output_dir = base_dir / "exp" / "omnivoice_vinv_lora"

    # Stage 0: Tokenize
    if stage <= 0 and stop_stage >= 0:
        print("\n========================================================")
        print(" Stage 0: Trích xuất Audio Codec Tokens (WebDataset)")
        print("========================================================")
        
        splits = [("train", train_jsonl), ("dev", dev_jsonl)]
        for split_name, jsonl_path in splits:
            if not jsonl_path.exists():
                print(f"Bỏ qua {split_name}, không tìm thấy: {jsonl_path}")
                continue
            
            split_out = token_dir / split_name
            split_out.mkdir(parents=True, exist_ok=True)
            tar_pattern = (split_out / "audios" / "shard-%06d.tar").as_posix()
            jsonl_pattern = (split_out / "txts" / "shard-%06d.jsonl").as_posix()

            cmd = [
                sys.executable, "-m", "omnivoice.scripts.extract_audio_tokens",
                "--input_jsonl", str(jsonl_path),
                "--tar_output_pattern", tar_pattern,
                "--jsonl_output_pattern", jsonl_pattern,
                "--tokenizer_path", tokenizer_path,
                "--nj_per_gpu", "1",
                "--shuffle", "True"
            ]
            run_command(cmd, env=current_env)
            print(f"-> Hoàn tất tokenizer cho {split_name}: {split_out / 'data.lst'}")

    # Stage 1: Fine-tune
    if stage <= 1 and stop_stage >= 1:
        print("\n========================================================")
        print(" Stage 1: Huấn luyện OmniVoice với LoRA")
        print("========================================================")
        cmd = [
            "accelerate", "launch",
            "--num_processes", "1",
            "-m", "omnivoice.cli.train",
            "--train_config", str(train_config),
            "--data_config", str(data_config),
            "--output_dir", str(output_dir)
        ]
        run_command(cmd, env=current_env)
        print(f"-> Hoàn thành fine-tuning OmniVoice! Model tại: {output_dir}")

if __name__ == "__main__":
    main()
