#!/usr/bin/env bash
set -euo pipefail

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo 'huggingface-cli is not installed. Run: pip install "huggingface_hub[cli]"' >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CKPT_DIR="${ROOT_DIR}/ckpts"
WAN_DIR="${CKPT_DIR}/Wan2.2-TI2V-5B"
MMAUDIO_TMP_DIR="${ROOT_DIR}/tmp/mmaudio_release"
MMAUDIO_EXT_DIR="${CKPT_DIR}/mmaudio/ext_weights"

mkdir -p "${WAN_DIR}" "${MMAUDIO_EXT_DIR}" "${MMAUDIO_TMP_DIR}"

echo "Downloading the minimal Wan files required by Foley-Omni..."
huggingface-cli download Wan-AI/Wan2.2-TI2V-5B   models_t5_umt5-xxl-enc-bf16.pth   google/umt5-xxl/special_tokens_map.json   google/umt5-xxl/spiece.model   google/umt5-xxl/tokenizer.json   google/umt5-xxl/tokenizer_config.json   --local-dir "${WAN_DIR}"

echo "Downloading the minimal MMAudio files required by Foley-Omni..."
huggingface-cli download hkchengrex/MMAudio   ext_weights/v1-16.pth   ext_weights/best_netG.pt   ext_weights/synchformer_state_dict.pth   --local-dir "${MMAUDIO_TMP_DIR}"

cp "${MMAUDIO_TMP_DIR}/ext_weights/"* "${MMAUDIO_EXT_DIR}/"

echo
echo "Done. Only the public dependency files required by Foley-Omni were downloaded to: ${CKPT_DIR}"
echo "You still need to download your released Foley-Omni checkpoint and place it under ckpts/Foley-Omni/v2st.pth"
