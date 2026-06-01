import json
import os
from collections import OrderedDict

import torch
from safetensors.torch import load_file

from foley_omni.modules.fusion import FusionModel
from foley_omni.modules.mmaudio.features_utils import FeaturesUtils
from foley_omni.modules.t5 import T5EncoderModel


def _first_existing_path(candidates):
    for path in candidates:
        if os.path.exists(path):
            return path
    return candidates[0]


def init_mmaudio_vae(ckpt_dir, rank=0, sample_rate=16000):
    """Initialize the MMAudio VAE used by the public inference path."""
    vae_config = {}

    if sample_rate == 16000:
        vae_config["mode"] = "16k"
        tod_vae_ckpt = os.path.join(ckpt_dir, "mmaudio/ext_weights/v1-16.pth")
    elif sample_rate == 44100:
        vae_config["mode"] = "44k"
        tod_vae_ckpt = os.path.join(ckpt_dir, "mmaudio/ext_weights/v1-44.pth")
    else:
        raise ValueError(f"Unsupported sample_rate: {sample_rate}. Only 16000 and 44100 are supported.")

    vae_config["need_vae_encoder"] = True
    vae_config["tod_vae_ckpt"] = tod_vae_ckpt
    vae_config["bigvgan_vocoder_ckpt"] = os.path.join(ckpt_dir, "mmaudio/ext_weights/best_netG.pt")

    return FeaturesUtils(**vae_config).to(rank)


def init_fusion_score_model(rank=0, meta_init=True, sample_rate=16000):
    """Initialize the released audio-only score model and its audio config."""
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))

    if sample_rate == 16000:
        audio_config_path = os.path.join(project_root, "foley_omni/configs/model/dit/audio.json")
    elif sample_rate == 44100:
        audio_config_path = os.path.join(project_root, "foley_omni/configs/model/dit/audio_44k.json")
    else:
        raise ValueError(f"Unsupported sample_rate: {sample_rate}. Only 16000 and 44100 are supported.")

    if not os.path.exists(audio_config_path):
        raise FileNotFoundError(f"Missing audio config: {audio_config_path}")

    with open(audio_config_path) as f:
        audio_config = json.load(f)

    if meta_init:
        with torch.device("meta"):
            fusion_model = FusionModel(video_config=None, audio_config=audio_config)
    else:
        fusion_model = FusionModel(video_config=None, audio_config=audio_config)

    params_all = sum(p.numel() for p in fusion_model.parameters())
    if rank == 0:
        print(f"Score model (audio-only) all parameters: {params_all}")

    return fusion_model, None, audio_config


def init_text_model(ckpt_dir, rank, cpu_offload=False):
    text_encoder_path = _first_existing_path([
        os.path.join(ckpt_dir, "models_t5_umt5-xxl-enc-bf16.pth"),
        os.path.join(ckpt_dir, "Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
    ])
    text_tokenizer_path = _first_existing_path([
        os.path.join(ckpt_dir, "google/umt5-xxl"),
        os.path.join(ckpt_dir, "Wan2.2-TI2V-5B/google/umt5-xxl"),
    ])

    return T5EncoderModel(
        text_len=512,
        dtype=torch.bfloat16,
        device=rank,
        checkpoint_path=text_encoder_path,
        tokenizer_path=text_tokenizer_path,
        cpu_offload=cpu_offload,
        shard_fn=None,
    )


def _extract_model_state_dict(ckpt_obj):
    """Extract the actual model state dict from common checkpoint layouts."""
    if not isinstance(ckpt_obj, dict):
        return ckpt_obj, "raw_non_dict"

    if "module" in ckpt_obj and isinstance(ckpt_obj["module"], dict):
        return ckpt_obj["module"], "module"
    if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
        return ckpt_obj["model"], "model"
    if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
        return ckpt_obj["state_dict"], "state_dict"
    if "app" in ckpt_obj and isinstance(ckpt_obj["app"], dict) and "model" in ckpt_obj["app"] and isinstance(ckpt_obj["app"]["model"], dict):
        return ckpt_obj["app"]["model"], "app.model"

    return ckpt_obj, "raw_dict"


def _best_key_transform(state_dict, model_keys_set):
    """Choose the key transform with the highest overlap to reduce false loads."""
    if not isinstance(state_dict, dict):
        return state_dict, "identity", 0

    candidates = [
        ("identity", lambda k: k),
        ("strip_module", lambda k: k[len("module."):] if k.startswith("module.") else k),
        ("strip_model", lambda k: k[len("model."):] if k.startswith("model.") else k),
        ("strip_state_dict", lambda k: k[len("state_dict."):] if k.startswith("state_dict.") else k),
    ]

    best_name = "identity"
    best_sd = state_dict
    best_overlap = -1

    for name, fn in candidates:
        remapped = OrderedDict()
        for key, value in state_dict.items():
            remapped[fn(key)] = value
        overlap = len(set(remapped.keys()) & model_keys_set)
        if overlap > best_overlap:
            best_overlap = overlap
            best_name = name
            best_sd = remapped

    return best_sd, best_name, max(best_overlap, 0)


def load_fusion_checkpoint(model, checkpoint_path, from_meta=False, verbose=True, key_print_limit=80):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        raise RuntimeError(f"checkpoint_path does not exist: {checkpoint_path}")

    if checkpoint_path.endswith(".safetensors"):
        ckpt_obj = load_file(checkpoint_path, device="cpu")
        source_name = "safetensors_raw"
    elif checkpoint_path.endswith(".pt") or checkpoint_path.endswith(".pth"):
        try:
            ckpt_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except Exception:
            ckpt_obj = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        ckpt_obj, source_name = _extract_model_state_dict(ckpt_obj)
    else:
        raise RuntimeError("Only .safetensors, .pt and .pth checkpoints are supported")

    if not isinstance(ckpt_obj, dict):
        raise RuntimeError(f"Checkpoint parsed from {checkpoint_path} is not a dict-like state_dict (got {type(ckpt_obj)}).")

    model_keys = list(model.state_dict().keys())
    model_keys_set = set(model_keys)

    state_dict, key_transform, _ = _best_key_transform(ckpt_obj, model_keys_set)
    checkpoint_keys = list(state_dict.keys())
    loaded_keys = [key for key in checkpoint_keys if key in model_keys_set]

    missing, unexpected = model.load_state_dict(state_dict, strict=False, assign=from_meta)

    if verbose:
        overlap_ratio = (len(loaded_keys) / max(len(model_keys), 1)) * 100.0
        print(f"[CKPT-DEBUG] checkpoint_path={checkpoint_path}")
        print(f"[CKPT-DEBUG] source={source_name}, key_transform={key_transform}")
        print(f"[CKPT-DEBUG] model_keys={len(model_keys)}, ckpt_keys={len(checkpoint_keys)}, overlap={len(loaded_keys)} ({overlap_ratio:.2f}%)")
        print(f"[CKPT-DEBUG] load_state_dict -> missing={len(missing)}, unexpected={len(unexpected)}")

        if loaded_keys:
            print(f"[CKPT-DEBUG] loaded_keys (showing up to {key_print_limit}):")
            for key in loaded_keys[:key_print_limit]:
                print(f"  + {key}")
            if len(loaded_keys) > key_print_limit:
                print(f"  ... ({len(loaded_keys) - key_print_limit} more)")
        else:
            print("[CKPT-DEBUG] loaded_keys is empty! This is highly likely a fake load / mismatched checkpoint.")

        if missing:
            print(f"[CKPT-DEBUG] missing_keys (showing up to {key_print_limit}):")
            for key in missing[:key_print_limit]:
                print(f"  - {key}")
            if len(missing) > key_print_limit:
                print(f"  ... ({len(missing) - key_print_limit} more)")

        if unexpected:
            print(f"[CKPT-DEBUG] unexpected_keys (showing up to {key_print_limit}):")
            for key in unexpected[:key_print_limit]:
                print(f"  ? {key}")
            if len(unexpected) > key_print_limit:
                print(f"  ... ({len(unexpected) - key_print_limit} more)")

        if overlap_ratio < 50.0:
            print(f"[CKPT-DEBUG][WARNING] Key overlap is only {overlap_ratio:.2f}%. Please verify checkpoint format/sample_rate/model config.")

    del ckpt_obj, state_dict
    import gc
    gc.collect()
    print(f"Successfully loaded fusion checkpoint from {checkpoint_path}")
    return {
        "model_keys": len(model_keys),
        "checkpoint_keys": len(checkpoint_keys),
        "loaded_keys": len(loaded_keys),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "source": source_name,
        "key_transform": key_transform,
    }
