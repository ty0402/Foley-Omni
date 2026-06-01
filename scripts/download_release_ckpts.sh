#!/usr/bin/env bash
set -euo pipefail

REPO_ID="${1:-}"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${REPO_ID}" ]]; then
  echo "Usage: bash scripts/download_release_ckpts.sh <org>/Foley-Omni" >&2
  exit 1
fi

if command -v hf >/dev/null 2>&1; then
  HF_CMD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_CMD=(huggingface-cli download)
else
  echo 'Neither `hf` nor `huggingface-cli` is installed. Run: pip install -U "huggingface_hub[cli]"' >&2
  exit 1
fi

"${HF_CMD[@]}" "${REPO_ID}" \
  ckpts/Foley-Omni/v2st.pth \
  ckpts/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
  ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/special_tokens_map.json \
  ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/spiece.model \
  ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer.json \
  ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/tokenizer_config.json \
  ckpts/mmaudio/ext_weights/v1-16.pth \
  ckpts/mmaudio/ext_weights/best_netG.pt \
  ckpts/mmaudio/ext_weights/synchformer_state_dict.pth \
  --local-dir "${ROOT_DIR}"

echo "Done. Checkpoints were downloaded under: ${ROOT_DIR}/ckpts"
echo "Note: the CLIP image encoder for online feature extraction is fetched by open_clip on first use."
