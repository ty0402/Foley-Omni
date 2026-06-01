# Foley-Omni

**Foley-Omni: A Unified Multimodal Generation Model from Task-Level Audio Synthesis to Complete Video Soundtrack Generation**

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv&logoColor=white)](ARXIV_LINK)
[![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-FFD21E?logo=huggingface&logoColor=000000)](https://huggingface.co/CocoBro/Foley-Omni)
[![Demo](https://img.shields.io/badge/Demo-Video-2563eb?logo=youtube&logoColor=white)](assets/demo.mp4)

# Demo

<p align="center">
  <video src="assets/demo.mp4" controls muted width="100%"></video>
</p>


# Overview



Foley-Omni focuses on **Video-to-Soundtrack (V2ST)** generation.
Given a video and optional text conditioning, Foley-Omni jointly generates synchronized **speech**, **sound effects**, and **music**. Besides, the model also supports single-task inference such as **Text-to-Audio (TTA)** and **Video-to-Audio (V2A)**.

> **V2ST-Bench** for complete video soundtrack generation：     **Coming soon** .

# Install

The public release was verified in the `video` environment with:

- Python 3.10
- CUDA 12.4
- PyTorch 2.6.0
- FlashAttention 2.7.4.post1

```bash
git clone CODE_REPO_LINK
cd foley-omni-blind

conda create -n foley-omni python=3.10 -y
conda activate foley-omni

# Install PyTorch first
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
pip install -r requirements.txt

# Install Flash Attention
pip install flash_attn==2.7.4.post1 --no-build-isolation

# Install the Hugging Face CLI
pip install -U "huggingface_hub[cli]"
```

---
## Download

The released checkpoints are hosted at `https://huggingface.co/CocoBro/Foley-Omni`.
Download the full checkpoint package with:

```bash
bash scripts/download_release_ckpts.sh CocoBro/Foley-Omni
```

Expected checkpoint layout:

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

# Inference

The current public  checkpoint is designed for videos **up to 10 seconds**.
For best results, trim each input video to **10 seconds or shorter** before inference.

## Run example

Batch inference:

```bash
python inference_v2st.py --config-file inference_v2st.yaml
```



Generated files will be written to `output_dir` and include:

- `*.mp4`: input video merged with the generated soundtrack


Single-video inference:

1. Disable `json_file` in [inference_v2st.yaml](/taoye/open/foley-omni-blind/inference_v2st.yaml)
2. Set `video_path`
3. Set `text_prompt`
4. Run:

```bash
python inference_v2st.py --config-file inference_v2st.yaml
```
## Format

The batch example file is:

- [examples/video_text_example.json](/taoye/open/foley-omni-blind/examples/video_text_example.json)

Each JSON key is a video path.
Each JSON value is a metadata object for soundtrack generation.

Minimal example:

```json
{
  "./examples/videos/721ecf7c92d162bd2d74820f72f68d41.mp4": {
    "resp": "[WORDS]That car came by faster than I expected.[END_WORDS][AUDIO_CAPTION]A clear, neutral English-speaking voice is accompanied by the sound of a car passing on a quiet urban street.[END_AUDIO_CAPTION]"
  }
}
```

Supported fields:

- `resp`: required structured prompt string
- `clip_feature_path`: optional pre-extracted CLIP feature path
- `sync_feature_path`: optional pre-extracted Sync feature path

The `resp` field can contain any subset of the following blocks:

- `[WORDS] ... [END_WORDS]`: speech content to be spoken in the generated soundtrack
- `[AUDIO_CAPTION] ... [END_AUDIO_CAPTION]`: sound effects, acoustic events, actions, speaker prompt
- `[MUSIC] ... [END_MUSIC]`: background music style, mood, instrumentation, and tempo

Notes:

- At least one of `WORDS`, `AUDIO_CAPTION`, or `MUSIC` should be present in each sample.
- `clip_feature_path` and `sync_feature_path` are optional.
- If feature paths are not provided, Foley-Omni extracts visual features from the input video.

## Prepare Visual Features

To pre-extract CLIP and Sync features, use:

- [data_process/convert_memmap_to_npy.py](/taoye/open/foley-omni-blind/data_process/convert_memmap_to_npy.py)

Example:

```bash
python data_process/convert_memmap_to_npy.py \
  --json_input ./examples/video_text_example.json \
  --feature_dir ./examples/features \
  --json_output ./examples/video_text_with_features.json \
  --gpu_ids 0
```

This script reads the input videos, extracts `clip_feature_path` and `sync_feature_path`, and writes an updated JSON manifest that can be used directly by [inference_v2st.py](/taoye/open/foley-omni-blind/inference_v2st.py).

## Text-Only Generation

Representative text-only prompts are provided at:

- [examples/text_example.jsonl](/taoye/open/foley-omni-blind/examples/text_example.jsonl)

The default text-only config is:

- [inference_fusion.yaml](/taoye/open/foley-omni-blind/inference_fusion.yaml)

Run text-only generation with:

```bash
python inference.py --config-file inference_fusion.yaml
```

# Acknowledgements

We would like to thank the following projects:

- **MMAudio**: Foley-Omni reuses MMAudio's audio VAE and feature extractor.
- **Ovi** and **Wan2.2**: the DiT design and implementation are primarily developed with reference to Ovi and Wan2.2.


