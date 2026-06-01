---
library_name: pytorch
license: other
pipeline_tag: text-to-audio
---

# Foley-Omni

**Foley-Omni: A Unified Multimodal Generation Model from Task-Level Audio Synthesis to Complete Video Soundtrack Generation**

[GitHub Code](CODE_REPO_LINK) | [arXiv](ARXIV_LINK) | [Demo](DEMO_LINK)

## Overview

This repository packages the public inference checkpoint set for **Foley-Omni**.
The release focuses on **Video-to-Soundtrack (V2ST)** generation, where the model jointly generates synchronized **speech**, **sound effects**, and **music** from a video and optional text prompt.

The main model checkpoint is provided as an inference-only release weight:

- `ckpts/Foley-Omni/v2st.pth`

## Repository Contents

```text
ckpts/
├── Foley-Omni/
│   └── v2st.pth
├── Wan2.2-TI2V-5B/
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── google/
│       └── umt5-xxl/
│           ├── special_tokens_map.json
│           ├── spiece.model
│           ├── tokenizer.json
│           └── tokenizer_config.json
└── mmaudio/
    └── ext_weights/
        ├── v1-16.pth
        ├── best_netG.pt
        └── synchformer_state_dict.pth
```

Checkpoint overview:

- `ckpts/Foley-Omni/v2st.pth`: released inference-only Foley-Omni weights
- `ckpts/Wan2.2-TI2V-5B/*`: text encoder and tokenizer for text conditioning
- `ckpts/mmaudio/ext_weights/v1-16.pth`: audio VAE for the 16 kHz inference path
- `ckpts/mmaudio/ext_weights/best_netG.pt`: vocoder for waveform decoding
- `ckpts/mmaudio/ext_weights/synchformer_state_dict.pth`: online visual feature extraction

## Online Feature Extraction

This release supports both:

- direct V2ST inference with pre-extracted `clip_feature_path` and `sync_feature_path`
- V2ST inference without pre-extracted features, using online visual feature extraction

Notes:

- `synchformer_state_dict.pth` is included because it is required for online Sync feature extraction.
- The CLIP image encoder is loaded by `open_clip` from `apple/DFN5B-CLIP-ViT-H-14-384` on first use. The current code path does not use a separate local CLIP checkpoint file.

## Source Attribution

This repository redistributes a small subset of files from the following upstream releases for convenience:

- **Wan2.2-TI2V-5B**: text encoder and tokenizer files
- **MMAudio**: audio VAE, vocoder, and Synchformer files

Please refer to the original upstream repositories for their licenses, usage terms, and project details.

## Acknowledgements

We would like to thank the following projects:

- **Wan2.2**: the video branch is initialized from the Wan2.2 repository.
- **MMAudio**: Foley-Omni reuses MMAudio's audio VAE and related audio decoding components.
- **Ovi** and **Wan2.2**: the DiT design and implementation are primarily developed with reference to Ovi and Wan2.2.

## Quick Start

Use the code repository for inference scripts, configs, examples, and feature extraction tools:

- `inference_v2st.py`
- `inference_v2st.yaml`
- `examples/video_text_example.json`
- `data_process/convert_memmap_to_npy.py`

Download the packaged checkpoints with:

```bash
hf download CocoBro/Foley-Omni \
  ckpts/Foley-Omni/v2st.pth \
  ckpts/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
  ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/special_tokens_map.json \
  ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/spiece.model \
  ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json \
  ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer_config.json \
  ckpts/mmaudio/ext_weights/v1-16.pth \
  ckpts/mmaudio/ext_weights/best_netG.pt \
  ckpts/mmaudio/ext_weights/synchformer_state_dict.pth \
  --local-dir .
```
