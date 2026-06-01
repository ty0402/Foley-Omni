# Hugging Face Release

## Overview

This document describes how to package the public Foley-Omni checkpoint release for Hugging Face.
The release script exports inference-only model weights and bundles the additional checkpoints required for text conditioning and online visual feature extraction.

## Recommended Repository Name

- `<org>/Foley-Omni`

## Why `hf upload-large-folder`

The original training checkpoint is large, so `hf upload-large-folder` is the safer option for resumable uploads.

## 1. Install the Hugging Face CLI

```bash
pip install -U "huggingface_hub[cli]"
```

## 2. Login

```bash
hf auth login
```

## 3. Create the model repository

```bash
hf repo create <org>/Foley-Omni --type model
```

Add `--private` if the repository should be uploaded privately first.

## 4. Stage the release files

```bash
bash scripts/stage_hf_release.sh
```

This creates:

```text
./hf_release
```

The staging script performs three tasks:

1. Exports inference-only Foley-Omni weights to `ckpts/Foley-Omni/v2st.pth`
2. Copies the Wan text encoder and tokenizer files used by inference
3. Copies the MMAudio VAE, vocoder, and Synchformer files required for waveform decoding and online visual feature extraction

The script searches common local locations for the dependency files and also supports explicit overrides through environment variables:

- `MODEL_CKPT_SRC`
- `WAN_T5_CKPT`
- `WAN_TOKENIZER_ROOT`
- `MMAUDIO_VAE_CKPT`
- `MMAUDIO_VOCODER_CKPT`
- `MMAUDIO_SYNCHFORMER_CKPT`

## 5. Edit the staged Hugging Face model card

Update the links in:

```text
./hf_release/README.md
```

Fill in:

- `CODE_REPO_LINK`
- `ARXIV_LINK`
- `DEMO_LINK`

Add the upstream attribution links for the bundled Wan2.2 and MMAudio files.

## 6. Upload to Hugging Face

```bash
hf upload-large-folder <org>/Foley-Omni ./hf_release
```

## 7. Verify the published layout

The published repository should contain:

```text
ckpts/
├── Foley-Omni/
│   └── v2st.pth
├── Wan2.2-TI2V-5B/
│   ├── models_t5_umt5-xxl-enc-bf16.pth
│   └── google/umt5-xxl/*
└── mmaudio/
    └── ext_weights/
        ├── v1-16.pth
        ├── best_netG.pt
        └── synchformer_state_dict.pth
README.md
```

## 8. Download command for end users

After publishing, the full checkpoint package can be downloaded with:

```bash
bash scripts/download_release_ckpts.sh <org>/Foley-Omni
```
