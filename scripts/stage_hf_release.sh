#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGE_DIR="${1:-${ROOT_DIR}/hf_release}"

MODEL_CKPT_SRC="${MODEL_CKPT_SRC:-/taoye/workspace/VR-Foley-5.5B-ty_debug/ckpts/v3_fintune_final/checkpoints/model-epoch-000010.pth}"
TORCH_PYTHON="${TORCH_PYTHON:-/taoye/miniconda3/envs/mmaudio/bin/python}"

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "Missing required file: $1" >&2
    exit 1
  fi
}

find_first_file() {
  local env_name="$1"
  shift
  local override="${!env_name:-}"
  if [[ -n "${override}" ]]; then
    if [[ -f "${override}" ]]; then
      printf '%s\n' "${override}"
      return 0
    fi
    echo "Configured path in ${env_name} does not exist: ${override}" >&2
    exit 1
  fi

  local candidate
  for candidate in "$@"; do
    if [[ -f "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

find_first_dir() {
  local env_name="$1"
  shift
  local override="${!env_name:-}"
  if [[ -n "${override}" ]]; then
    if [[ -d "${override}" ]]; then
      printf '%s\n' "${override}"
      return 0
    fi
    echo "Configured path in ${env_name} does not exist: ${override}" >&2
    exit 1
  fi

  local candidate
  for candidate in "$@"; do
    if [[ -d "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  return 1
}

require_file "${MODEL_CKPT_SRC}"

WAN_T5_CKPT="$(find_first_file WAN_T5_CKPT \
  /taoye/ty/data/ckpt/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth \
  /taoye/ty/data/ckpt/models_t5_umt5-xxl-enc-bf16.pth)"
WAN_TOKENIZER_ROOT="$(find_first_dir WAN_TOKENIZER_ROOT \
  /taoye/ty/data/ckpt/Wan2.2-TI2V-5B/google/umt5-xxl \
  /taoye/ty/data/ckpt/google/umt5-xxl)"
MMAUDIO_VAE_CKPT="$(find_first_file MMAUDIO_VAE_CKPT \
  /taoye/ty/data/ckpt/MMAudio/ext_weights/v1-16.pth \
  /taoye/workspace/VRSound/ext_weights/v1-16.pth)"
MMAUDIO_VOCODER_CKPT="$(find_first_file MMAUDIO_VOCODER_CKPT \
  /taoye/ty/data/ckpt/MMAudio/ext_weights/best_netG.pt \
  /taoye/workspace/VRSound/ext_weights/best_netG.pt)"
MMAUDIO_SYNCHFORMER_CKPT="$(find_first_file MMAUDIO_SYNCHFORMER_CKPT \
  /taoye/workspace/VRSound/ext_weights/synchformer_state_dict.pth \
  /taoye/ty/data/ckpt/MMAudio/ext_weights/synchformer_state_dict.pth)"

require_file "${WAN_T5_CKPT}"
require_file "${WAN_TOKENIZER_ROOT}/special_tokens_map.json"
require_file "${WAN_TOKENIZER_ROOT}/spiece.model"
require_file "${WAN_TOKENIZER_ROOT}/tokenizer.json"
require_file "${WAN_TOKENIZER_ROOT}/tokenizer_config.json"
require_file "${MMAUDIO_VAE_CKPT}"
require_file "${MMAUDIO_VOCODER_CKPT}"
require_file "${MMAUDIO_SYNCHFORMER_CKPT}"

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/ckpts/Foley-Omni"
mkdir -p "${STAGE_DIR}/ckpts/Wan2.2-TI2V-5B/google/umt5-xxl"
mkdir -p "${STAGE_DIR}/ckpts/mmaudio/ext_weights"

"${TORCH_PYTHON}" "${ROOT_DIR}/scripts/extract_inference_checkpoint.py" \
  "${MODEL_CKPT_SRC}" \
  "${STAGE_DIR}/ckpts/Foley-Omni/v2st.pth" \
  --print-summary

cp "${WAN_T5_CKPT}" "${STAGE_DIR}/ckpts/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"
cp "${WAN_TOKENIZER_ROOT}/special_tokens_map.json" "${STAGE_DIR}/ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/"
cp "${WAN_TOKENIZER_ROOT}/spiece.model" "${STAGE_DIR}/ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/"
cp "${WAN_TOKENIZER_ROOT}/tokenizer.json" "${STAGE_DIR}/ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/"
cp "${WAN_TOKENIZER_ROOT}/tokenizer_config.json" "${STAGE_DIR}/ckpts/Wan2.2-TI2V-5B/google/umt5-xxl/"
cp "${MMAUDIO_VAE_CKPT}" "${STAGE_DIR}/ckpts/mmaudio/ext_weights/v1-16.pth"
cp "${MMAUDIO_VOCODER_CKPT}" "${STAGE_DIR}/ckpts/mmaudio/ext_weights/best_netG.pt"
cp "${MMAUDIO_SYNCHFORMER_CKPT}" "${STAGE_DIR}/ckpts/mmaudio/ext_weights/synchformer_state_dict.pth"
cp "${ROOT_DIR}/HF_MODEL_CARD.md" "${STAGE_DIR}/README.md"

echo "Staged Hugging Face release at: ${STAGE_DIR}"
echo "Resolved source files:"
echo "  Foley-Omni training checkpoint: ${MODEL_CKPT_SRC}"
echo "  Wan T5 checkpoint: ${WAN_T5_CKPT}"
echo "  Wan tokenizer root: ${WAN_TOKENIZER_ROOT}"
echo "  MMAudio VAE: ${MMAUDIO_VAE_CKPT}"
echo "  MMAudio vocoder: ${MMAUDIO_VOCODER_CKPT}"
echo "  MMAudio Synchformer: ${MMAUDIO_SYNCHFORMER_CKPT}"
echo "Next steps:"
echo "  1. Edit ${STAGE_DIR}/README.md and fill the links."
echo "  2. Create or login to the target model repo."
echo "  3. Upload with: hf upload-large-folder <your-org>/Foley-Omni ${STAGE_DIR}"
