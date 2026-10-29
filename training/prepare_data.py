import os
import json
import glob
from pathlib import Path

def prepare_omnivoice_and_vits():
    base_dir = Path(__file__).resolve().parent
    vinv_dir = base_dir / "vinv-tts"
    
    # Destination directories
    omnivoice_data_dir = base_dir / "data" / "omnivoice"
    vits_data_dir = base_dir / "data" / "vits"
    omnivoice_data_dir.mkdir(parents=True, exist_ok=True)
    vits_data_dir.mkdir(parents=True, exist_ok=True)

    splits = ["train", "dev"]
    
    stats = {}

    for split in splits:
        split_dir = vinv_dir / split
        if not split_dir.exists():
            print(f"Warning: Directory {split_dir} does not exist!")
            continue

        json_files = sorted(list(split_dir.glob("spk_*/*.json")))
        print(f"Processing split '{split}': Found {len(json_files)} speaker JSON files.")

        omni_records = []
        vits_records = []

        total_audio = 0
        tag_counts = {"[breathing]": 0, "[laughter]": 0, "[sniff]": 0, "[throatclearing]": 0}

        for jf in json_files:
            spk_id = jf.parent.name
            try:
                with open(jf, "r", encoding="utf-8") as f:
                    items = json.load(f)
            except Exception as e:
                print(f"Error loading {jf}: {e}")
                continue

            for item in items:
                audio_rel = item.get("audio")
                audio_path = (jf.parent / audio_rel).resolve().as_posix()
                text = item.get("text", "").strip()
                lang = item.get("language_id", "vi")

                if not os.path.exists(audio_path):
                    continue

                for tag in tag_counts:
                    if tag in text:
                        tag_counts[tag] += text.count(tag)

                # OmniVoice record
                sample_id = f"{split}_{spk_id}_{Path(audio_rel).stem}"
                omni_records.append({
                    "id": sample_id,
                    "audio_path": audio_path,
                    "text": text,
                    "language_id": lang
                })

                # VITS record (audio_path|speaker_id|text)
                vits_records.append(f"{audio_path}|{spk_id}|{text}")
                total_audio += 1

        # Write OmniVoice JSONL
        omni_out_path = omnivoice_data_dir / f"{split}.jsonl"
        with open(omni_out_path, "w", encoding="utf-8") as f:
            for r in omni_records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # Write VITS metadata
        vits_out_path = vits_data_dir / f"{split}.txt"
        with open(vits_out_path, "w", encoding="utf-8") as f:
            for line in vits_records:
                f.write(line + "\n")

        stats[split] = {
            "speakers": len(json_files),
            "samples": total_audio,
            "omni_manifest": str(omni_out_path),
            "vits_manifest": str(vits_out_path),
            "tag_counts": tag_counts
        }

    print("\n--- Summary of prepared data ---")
    print(json.dumps(stats, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    prepare_omnivoice_and_vits()
