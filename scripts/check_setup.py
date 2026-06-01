#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

TOKENIZER_FILES = [
    "special_tokens_map.json",
    "spiece.model",
    "tokenizer.json",
    "tokenizer_config.json",
]


def main():
    parser = argparse.ArgumentParser(description="Check required files for the current V2ST public release.")
    parser.add_argument("--ckpt-dir", type=Path, default=Path("./ckpts"))
    parser.add_argument("--model-checkpoint", type=Path, default=Path("./ckpts/Foley-Omni/v2st.pth"))
    parser.add_argument("--preextracted-features", action="store_true")
    args = parser.parse_args()

    checks = [
        (args.model_checkpoint, "Released Foley-Omni inference checkpoint"),
        (args.ckpt_dir / "Wan2.2-TI2V-5B" / "models_t5_umt5-xxl-enc-bf16.pth", "Wan T5 encoder"),
        (args.ckpt_dir / "mmaudio" / "ext_weights" / "v1-16.pth", "MMAudio 16 kHz audio VAE"),
        (args.ckpt_dir / "mmaudio" / "ext_weights" / "best_netG.pt", "MMAudio vocoder"),
    ]

    tokenizer_root = args.ckpt_dir / "Wan2.2-TI2V-5B" / "google" / "umt5-xxl"
    for name in TOKENIZER_FILES:
        checks.append((tokenizer_root / name, f"Wan tokenizer file: {name}"))

    if not args.preextracted_features:
        checks.append((args.ckpt_dir / "mmaudio" / "ext_weights" / "synchformer_state_dict.pth", "Synchformer for online feature extraction"))

    missing = []
    print("[Setup Check] Required files for the current public V2ST path:")
    for path, reason in checks:
        ok = path.exists()
        status = "OK" if ok else "MISSING"
        print(f"- [{status}] {path} :: {reason}")
        if not ok:
            missing.append((path, reason))

    print("\n[Setup Check] Note: the CLIP image encoder used by online feature extraction is downloaded by open_clip on first use.")

    if missing:
        print(f"\n[Setup Check] Missing {len(missing)} required file(s).", file=sys.stderr)
        sys.exit(1)

    print("\n[Setup Check] All required local files are present.")


if __name__ == "__main__":
    main()
