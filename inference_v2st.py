import argparse
import json
import os
import sys
import logging
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from omegaconf import OmegaConf
from foley_omni.utils.io_utils import save_video
from foley_omni.distributed_comms.util import get_world_size, get_local_rank, get_global_rank
from foley_omni.distributed_comms.parallel_states import initialize_sequence_parallel_state, get_sequence_parallel_state, nccl_info
from foley_omni.fusion_engine import FoleyOmniEngine

MIN_SYNC_SEGMENT_FRAMES = 16

try:
    sys.path.insert(0, str(Path(__file__).parent / "mmaudio"))
    from mmaudio.eval_utils import load_video, all_model_cfg, make_video
    from mmaudio.model.utils.features_utils import FeaturesUtils
    from mmaudio.data.av_utils import VideoInfo
    from fractions import Fraction
    MMAUDIO_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Failed to import mmaudio modules: {e}")
    MMAUDIO_AVAILABLE = False


class SkipShortVideoError(Exception):
    """Raised when video is too short for sync feature extraction."""


def _make_zero_feature_like(feature, feature_kind, *, device, dtype):
    if feature is not None:
        return torch.zeros_like(feature, device=device, dtype=dtype)
    if feature_kind == "clip":
        return torch.zeros((80, 1024), device=device, dtype=dtype)
    if feature_kind == "sync":
        return torch.zeros((240, 768), device=device, dtype=dtype)
    raise ValueError(f"Unsupported feature_kind: {feature_kind}")


def _resolve_negative_video_feature_cfg(config):
    """
    Resolve whether CFG should act on each video feature in the negative branch.

    Backward compatibility:
    - legacy `cfg_zero_video_features_in_negative=true` means CFG *does* act on both
      clip/sync features because the negative branch removes them.
    - legacy per-feature zero switches are treated the same way.
    - new `cfg_apply_to_clip_features` / `cfg_apply_to_sync_features` override the
      legacy behavior explicitly.
    """
    legacy_default = bool(config.get("cfg_zero_video_features_in_negative", True))
    legacy_clip = bool(config.get("cfg_zero_clip_features_in_negative", legacy_default))
    legacy_sync = bool(config.get("cfg_zero_sync_features_in_negative", legacy_default))
    apply_clip = bool(config.get("cfg_apply_to_clip_features", legacy_clip))
    apply_sync = bool(config.get("cfg_apply_to_sync_features", legacy_sync))
    return apply_clip, apply_sync


def append_jsonl_record(jsonl_path, record):
    """Append one JSON line atomically (best effort across multi-process workers)."""
    line = json.dumps(record, ensure_ascii=False) + "\n"
    fd = os.open(jsonl_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, line.encode("utf-8"))
    finally:
        os.close(fd)


def _init_logging(rank):
    # logging
    if rank == 0:
        # set format
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)])
    else:
        logging.basicConfig(level=logging.ERROR)


def get_video_duration_accurate(video_path):
    """Read video duration by decoding frames and dividing by frame rate."""
    try:
        import av
        with av.open(video_path) as container:
            video_stream = container.streams.video[0]
            
            video_fps = video_stream.average_rate
            if video_fps is None:
                video_fps = video_stream.guessed_rate
            if video_fps is None:
                video_fps = 24.0
            
            if hasattr(video_fps, 'numerator') and hasattr(video_fps, 'denominator'):
                video_fps = float(video_fps.numerator) / float(video_fps.denominator)
            else:
                video_fps = float(video_fps)
            
            frame_count = 0
            for frame in container.decode(video_stream):
                frame_count += 1
            
            if frame_count == 0:
                logging.warning("[get_video_duration_accurate] Could not decode video frames; returning None")
                return None, None
            
            duration = frame_count / video_fps
            logging.info(f"[get_video_duration_accurate] Decoded {frame_count} frames at {video_fps:.2f} fps; duration={duration:.2f}s")
            return duration, video_fps
            
    except Exception as e:
        logging.warning(f"[get_video_duration_accurate] Failed to read video duration: {e}")
        return None, None


def calculate_audio_latent_length(duration):
    """Estimate the audio latent length from video duration."""
    audio_latent_length = int(round(duration * 31.4))
    return audio_latent_length


def load_video_features(video_path, clip_features_path=None, sync_features_path=None, 
                        duration=10.0, device='cuda', dtype=torch.bfloat16):
    """Load CLIP/Sync features from disk or extract them from the input video."""
    if not MMAUDIO_AVAILABLE:
        raise ImportError("mmaudio modules are unavailable, so video features cannot be extracted.")
    
    if clip_features_path and os.path.exists(clip_features_path):
        logging.info(f"Loading CLIP features from file: {clip_features_path}")
        clip_features = torch.from_numpy(np.load(clip_features_path)).to(device, dtype)
        if clip_features.ndim == 3:
            clip_features = clip_features.squeeze(0)
    else:
        clip_features = None
    
    if sync_features_path and os.path.exists(sync_features_path):
        logging.info(f"Loading Sync features from file: {sync_features_path}")
        sync_features = torch.from_numpy(np.load(sync_features_path)).to(device, dtype)
        if sync_features.ndim == 3:
            sync_features = sync_features.squeeze(0)
    else:
        sync_features = None
    
    if clip_features is None or sync_features is None:
        logging.info(f"Extracting features from video: {video_path}")
        
        actual_duration = duration
        accurate_duration, _ = get_video_duration_accurate(video_path)
        if accurate_duration is not None and accurate_duration > 0:
            actual_duration = accurate_duration
            logging.info(f"[load_video_features] Using actual video duration {actual_duration:.2f}s (requested {duration:.2f}s)")
        else:
            logging.warning(f"[load_video_features] Could not read accurate video duration; using requested duration {duration:.2f}s")
        
        video_info = load_video(Path(video_path), actual_duration)
        logging.info(f"Video duration: {video_info.duration_sec:.2f}s")
        logging.info(f"CLIP frame count: {video_info.clip_frames.shape[0]}")
        logging.info(f"Sync frame count: {video_info.sync_frames.shape[0]}")
        sync_frame_count = int(video_info.sync_frames.shape[0])
        if sync_frame_count < MIN_SYNC_SEGMENT_FRAMES:
            raise SkipShortVideoError(
                f"Video is too short for sync features: {sync_frame_count} < {MIN_SYNC_SEGMENT_FRAMES}, video={video_path}"
            )
        
        variant = 'large_44k_v2'
        model = all_model_cfg[variant]
        
        def resolve_path(path):
            if path.exists():
                return path
            return path
        
        synchformer_ckpt = resolve_path(model.synchformer_ckpt)

        # Online visual feature extraction only needs CLIP and Synchformer.
        # The audio VAE and vocoder are not required for clip/sync encoding.
        feature_utils = FeaturesUtils(
            tod_vae_ckpt=None,
            synchformer_ckpt=str(synchformer_ckpt),
            enable_conditions=True,
            mode=model.mode,
            bigvgan_vocoder_ckpt=None,
            need_vae_encoder=False
        )
        feature_utils = feature_utils.to(device, dtype).eval()
        
        clip_frames = video_info.clip_frames.unsqueeze(0).to(device, dtype)  # (1, T, C, H, W)
        sync_frames = video_info.sync_frames.unsqueeze(0).to(device, dtype)  # (1, T, C, H, W)
        
        if clip_features is None:
            logging.info("Extracting CLIP features...")
            with torch.inference_mode():
                clip_features = feature_utils.encode_video_with_clip(clip_frames)
            clip_features = clip_features.squeeze(0)  # (T_clip, 1024)
            logging.info(f"CLIP feature shape: {clip_features.shape}")
        
        if sync_features is None:
            logging.info("Extracting Sync features...")
            with torch.inference_mode():
                sync_features = feature_utils.encode_video_with_sync(sync_frames)
            sync_features = sync_features.squeeze(0)  # (T_sync, 768)
            logging.info(f"Sync feature shape: {sync_features.shape}")
    
    return clip_features, sync_features


def generate_audio_for_video(
    foley_omni_engine,
    video_path,
    text_prompt,
    clip_features_path=None,
    sync_features_path=None,
    seed=100,
    solver_name="unipc",
    sample_steps=50,
    shift=5.0,
    audio_guidance_scale=3.0,
    slg_layer=11,
    audio_negative_prompt="",
    duration=10.0,
    cfg_zero_video_features_in_negative=True,
    cfg_zero_clip_features_in_negative=None,
    cfg_zero_sync_features_in_negative=None,
    cfg_apply_to_clip_features=None,
    cfg_apply_to_sync_features=None,
    device='cuda',
    dtype=torch.bfloat16
):
    """Generate soundtrack audio for a single input video."""
    if abs(duration - 10.0) < 0.01:
        accurate_duration, _ = get_video_duration_accurate(video_path)
        if accurate_duration is not None and accurate_duration > 0 and abs(accurate_duration - 10.0) > 0.1:
            duration = accurate_duration
            logging.info(f"[generate_audio_for_video] Duration looked like a default value; re-read video duration as {duration:.2f}s")
    
    logging.info(f"[generate_audio_for_video] Starting audio generation with duration={duration:.2f}s")
    
    clip_features, sync_features = load_video_features(
        video_path, 
        clip_features_path, 
        sync_features_path,
        duration=duration,
        device=device,
        dtype=dtype
    )
    
    
    model = foley_omni_engine.model
    device = foley_omni_engine.device
    target_dtype = foley_omni_engine.target_dtype

    if cfg_zero_clip_features_in_negative is None:
        cfg_zero_clip_features_in_negative = cfg_zero_video_features_in_negative
    if cfg_zero_sync_features_in_negative is None:
        cfg_zero_sync_features_in_negative = cfg_zero_video_features_in_negative
    if cfg_apply_to_clip_features is None:
        cfg_apply_to_clip_features = cfg_zero_clip_features_in_negative
    if cfg_apply_to_sync_features is None:
        cfg_apply_to_sync_features = cfg_zero_sync_features_in_negative

    logging.info(
        "[CFG] apply to video features: clip=%s, sync=%s",
        cfg_apply_to_clip_features,
        cfg_apply_to_sync_features,
    )
    
    if foley_omni_engine.cpu_offload:
        foley_omni_engine.text_model.model = foley_omni_engine.text_model.model.to(device)
    
    text_embeddings = foley_omni_engine.text_model(
        [text_prompt, "", audio_negative_prompt], 
        foley_omni_engine.text_model.device
    )
    text_embeddings = [emb.to(target_dtype).to(device) for emb in text_embeddings]
    
    if foley_omni_engine.cpu_offload:
        foley_omni_engine.offload_to_cpu(foley_omni_engine.text_model.model)
    
    text_embeddings_audio_pos = text_embeddings[0]
    text_embeddings_audio_neg = text_embeddings[2]
    
    base_duration = 10.0
    if hasattr(foley_omni_engine, "config"):
        try:
            base_duration = float(foley_omni_engine.config.get("duration", base_duration))
        except Exception:
            pass
    latents_per_second = foley_omni_engine.audio_latent_length / max(base_duration, 1e-6)
    audio_latent_length = max(int(round(duration * latents_per_second)), 1)

    sample_rate = 16000
    if hasattr(foley_omni_engine, "config"):
        try:
            sample_rate = int(foley_omni_engine.config.get("sample_rate", sample_rate))
        except Exception:
            pass

    logging.info("=== Audio length calculation ===")
    logging.info(f"Video duration: {duration:.2f}s")
    logging.info(f"Computed audio_latent_length: {audio_latent_length}")
    logging.info(f"Base FoleyOmniEngine audio_latent_length: {foley_omni_engine.audio_latent_length}")
    logging.info(f"Target final audio length: {int(duration * sample_rate)} samples ({duration:.2f}s at {sample_rate} Hz)")
    
    audio_noise = torch.randn(
        (audio_latent_length, foley_omni_engine.audio_latent_channel),
        device=device,
        dtype=target_dtype,
        generator=torch.Generator(device=device).manual_seed(seed)
    )
    logging.info(f"Initial audio_noise shape: {audio_noise.shape}")
    
    max_seq_len_audio = audio_noise.shape[0]
    
    scheduler_audio, timesteps_audio = foley_omni_engine.get_scheduler_time_steps(
        sampling_steps=sample_steps,
        device=device,
        solver_name=solver_name,
        shift=shift
    )
    
    clip_features_list = [clip_features] if clip_features is not None else None
    sync_features_list = [sync_features] if sync_features is not None else None
    neg_clip_features_list = clip_features_list
    neg_sync_features_list = sync_features_list
    if cfg_apply_to_clip_features:
        neg_clip_features_list = [
            _make_zero_feature_like(clip_features, "clip", device=device, dtype=target_dtype)
        ]
    if cfg_apply_to_sync_features:
        neg_sync_features_list = [
            _make_zero_feature_like(sync_features, "sync", device=device, dtype=target_dtype)
        ]
    
    if foley_omni_engine.cpu_offload:
        foley_omni_engine.offload_to_cpu(foley_omni_engine.vae_model_audio)
        model = model.to(device)
    
    model.eval()
    with torch.inference_mode():
        with torch.amp.autocast('cuda', enabled=target_dtype != torch.float32, dtype=target_dtype):
            for i, t_a in tqdm(enumerate(timesteps_audio), desc="Sampling"):
                timestep_input = torch.full((1,), t_a, device=device)
                
                # Positive (conditional) forward pass
                pos_forward_args = {
                    'audio_context': [text_embeddings_audio_pos],
                    'vid_context': None,
                    'vid_seq_len': None,
                    'audio_seq_len': max_seq_len_audio,
                    'first_frame_is_clean': False,
                    'clip_features': clip_features_list,
                    'sync_features': sync_features_list
                }
                
                pred_vid_pos, pred_audio_pos = model(
                    vid=None,
                    audio=[audio_noise],
                    t=timestep_input,
                    **pos_forward_args
                )
                
                # Negative (unconditional) forward pass
                neg_forward_args = {
                    'audio_context': [text_embeddings_audio_neg],
                    'vid_context': None,
                    'vid_seq_len': None,
                    'audio_seq_len': max_seq_len_audio,
                    'first_frame_is_clean': False,
                    'slg_layer': slg_layer,
                    'clip_features': neg_clip_features_list,
                    'sync_features': neg_sync_features_list
                }
                
                pred_vid_neg, pred_audio_neg = model(
                    vid=None,
                    audio=[audio_noise],
                    t=timestep_input,
                    **neg_forward_args
                )
                
                # Apply classifier-free guidance
                pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])
                
                # Update noise using scheduler
                audio_noise_updated = scheduler_audio.step(
                    pred_audio_guided.unsqueeze(0), t_a, audio_noise.unsqueeze(0), return_dict=False
                )[0].squeeze(0)
                if audio_noise_updated.shape[0] != audio_noise.shape[0]:
                    logging.warning(f"Warning: scheduler.step changed audio length {audio_noise.shape[0]} -> {audio_noise_updated.shape[0]}")
                audio_noise = audio_noise_updated
                
                del pred_vid_pos, pred_audio_pos, pred_vid_neg, pred_audio_neg, pred_audio_guided, timestep_input
                if i % 5 == 0:
                    torch.cuda.empty_cache()
        
        if foley_omni_engine.cpu_offload:
            foley_omni_engine.offload_to_cpu(model)
            foley_omni_engine.vae_model_audio = foley_omni_engine.vae_model_audio.to(device)
        
        # Decode audio
        logging.info(f"Final audio_noise shape after sampling: {audio_noise.shape}")
        audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)  # 1, c, l
        logging.info(f"Audio latent shape before decoding: {audio_latents_for_vae.shape}")
        generated_audio = foley_omni_engine.vae_model_audio.wrapped_decode(audio_latents_for_vae)
        generated_audio = generated_audio.squeeze().cpu().float().numpy()
        logging.info(f"Raw decoded audio length: {len(generated_audio)} samples ({len(generated_audio)/sample_rate:.2f}s at {sample_rate} Hz)")
        
        target_audio_length = int(duration * sample_rate)
        current_audio_length = len(generated_audio)
        logging.info(f"Target audio length from video duration: {target_audio_length} samples ({duration:.2f}s)")
        
        if current_audio_length != target_audio_length:
            delta = current_audio_length - target_audio_length
            delta_sec = abs(delta) / float(sample_rate)
            logging.info(
                f"[Audio length adjust: stage 1] before: current={current_audio_length} ({current_audio_length/float(sample_rate):.3f}s), "
                f"target={target_audio_length} ({target_audio_length/float(sample_rate):.3f}s), "
                f"delta={delta} samples ({delta_sec:.3f}s)"
            )
            if current_audio_length > target_audio_length:
                generated_audio = generated_audio[:target_audio_length]
                logging.info(f"[Audio length adjust: stage 1] trimmed audio to match video duration: {current_audio_length} -> {target_audio_length} samples")
            else:
                padding = target_audio_length - current_audio_length
                generated_audio = np.pad(generated_audio, (0, padding), mode='constant')
                logging.info(f"[Audio length adjust: stage 1] padded audio to match video duration: {current_audio_length} -> {target_audio_length} samples (padding {padding} samples)")
            logging.info(
                f"[Audio length adjust: stage 1] after: final={len(generated_audio)} ({len(generated_audio)/float(sample_rate):.3f}s)"
            )
        else:
            logging.info("[Audio length adjust] Audio length already matches video duration")
        
        del audio_noise, audio_latents_for_vae
        if foley_omni_engine.cpu_offload:
            foley_omni_engine.offload_to_cpu(foley_omni_engine.vae_model_audio)
        
        del text_embeddings, text_embeddings_audio_pos, text_embeddings_audio_neg
        del clip_features, sync_features, clip_features_list, sync_features_list
        del scheduler_audio, timesteps_audio
        torch.cuda.empty_cache()
    
    return generated_audio


def main(config, args):
    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = local_rank
    torch.cuda.set_device(local_rank)
    sp_size = config.get("sp_size", 1)
    assert sp_size <= world_size and world_size % sp_size == 0, "sp_size must be less than or equal to world_size and world_size must be divisible by sp_size."

    _init_logging(global_rank)

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size)
    else:
        assert sp_size == 1, f"When world_size is 1, sp_size must also be 1, but got {sp_size}."

    initialize_sequence_parallel_state(sp_size)
    logging.info(f"Using SP: {get_sequence_parallel_state()}, SP_SIZE: {sp_size}")
    
    args.local_rank = local_rank
    args.device = device
    target_dtype = torch.bfloat16

    json_file = config.get("json_file", None)
    force_empty_text = bool(config.get("force_empty_text", False))
    
    if json_file is not None:
        logging.info(f"Loading JSON data from: {json_file}")
        assert os.path.isfile(json_file), f"JSON file not found: {json_file}"
        
        import json
        with open(json_file, 'r', encoding='utf-8') as f:
            json_data = json.load(f)
        
        all_eval_data = []
        for video_path, video_info in json_data.items():
            if not os.path.isfile(video_path):
                logging.warning(f"Video file not found; skipping: {video_path}")
                continue
            
            text_prompt = video_info.get("resp", "")
            if force_empty_text:
                text_prompt = ""
            clip_features_path = video_info.get("clip_feature_path", None)
            sync_features_path = video_info.get("sync_feature_path", None)
            
            if clip_features_path and not os.path.isfile(clip_features_path):
                logging.warning(f"CLIP feature file not found: {clip_features_path}; features will be extracted from video")
                clip_features_path = None
            if sync_features_path and not os.path.isfile(sync_features_path):
                logging.warning(f"Sync feature file not found: {sync_features_path}; features will be extracted from video")
                sync_features_path = None
            
            all_eval_data.append({
                'video_path': video_path,
                'text_prompt': text_prompt,
                'clip_features_path': clip_features_path,
                'sync_features_path': sync_features_path
            })
        
        logging.info(f"Loaded {len(all_eval_data)} videos from JSON")
        
    else:
        video_path = config.get("video_path")
        text_prompt = config.get("text_prompt", "")
        clip_features_path = config.get("clip_features_path", None)
        sync_features_path = config.get("sync_features_path", None)
        
        if video_path is not None:
            assert os.path.isfile(video_path), f"Video file not found: {video_path}"
            
            if isinstance(text_prompt, str) and text_prompt.endswith('.csv'):
                import pandas as pd
                df = pd.read_csv(text_prompt)
                if 'prompt' in df.columns:
                    text_prompts = df['prompt'].tolist()
                else:
                    text_prompts = [text_prompt] * len(df)
            else:
                text_prompts = [text_prompt] if text_prompt else [""]
            
            all_eval_data = []
            for text_p in text_prompts:
                all_eval_data.append({
                    'video_path': video_path,
                    'text_prompt': text_p,
                    'clip_features_path': clip_features_path,
                    'sync_features_path': sync_features_path
                })
        else:
            logging.error("Either json_file or video_path must be provided")
            return

    logging.info("Loading FoleyOmni engine...")
    foley_omni_engine = FoleyOmniEngine(config=config, device=device, target_dtype=target_dtype)
    logging.info("FoleyOmni engine loaded!")
    
    output_dir = config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    pred_map_jsonl_path = config.get("pred_map_jsonl", os.path.join(output_dir, "pred_mapping.jsonl"))

    if global_rank == 0:
        try:
            config_save_path = os.path.join(output_dir, "inference_config.yaml")
            OmegaConf.save(config=config, f=config_save_path)
            logging.info(f"Saved inference config to: {config_save_path}")
            logging.info(f"Prediction mapping JSONL will be written to: {pred_map_jsonl_path}")
        except Exception as e:
            logging.warning(f"Could not save inference config to output directory: {e}")

    # Get SP configuration
    use_sp = get_sequence_parallel_state()
    if use_sp:
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
    else:
        sp_size = 1
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    # Data distribution - by SP groups
    total_files = len(all_eval_data)
    
    if total_files == 0:
        logging.error(f"ERROR: No evaluation files found")
        this_rank_eval_data = []
    else:
        # Distribute across SP groups
        this_rank_eval_data = all_eval_data[sp_group_id :: num_sp_groups]

    for _, eval_item in tqdm(enumerate(this_rank_eval_data), desc="Processing videos"):
        video_path = eval_item['video_path']
        text_prompt = eval_item['text_prompt']
        video_clip_features_path = eval_item.get('clip_features_path', None)
        video_sync_features_path = eval_item.get('sync_features_path', None)
        
        seed = config.get("seed", 100)
        solver_name = config.get("solver_name", "unipc")
        sample_steps = config.get("sample_steps", 50)
        shift = config.get("shift", 5.0)
        audio_guidance_scale = config.get("audio_guidance_scale", 3.0)
        slg_layer = config.get("slg_layer", 11)
        audio_negative_prompt = config.get("audio_negative_prompt", "")
        cfg_zero_video_features_in_negative = bool(config.get("cfg_zero_video_features_in_negative", True))
        cfg_apply_to_clip_features, cfg_apply_to_sync_features = (
            _resolve_negative_video_feature_cfg(config)
        )
        duration = config.get("duration", 10.0)
        sample_rate = int(config.get("sample_rate", 16000))
        audio_only = bool(config.get("audio_only", False))
        
        if video_clip_features_path is None or video_sync_features_path is None:
            video_stem = Path(video_path).stem
            possible_dirs = [
                os.path.dirname(video_path),
                output_dir,
                os.path.join(os.path.dirname(video_path), "features"),
            ]
            for dir_path in possible_dirs:
                if os.path.exists(dir_path):
                    clip_path = os.path.join(dir_path, f"{video_stem}_clip_features.npy")
                    sync_path = os.path.join(dir_path, f"{video_stem}_sync_features.npy")
                    if video_clip_features_path is None and os.path.exists(clip_path):
                        video_clip_features_path = clip_path
                    if video_sync_features_path is None and os.path.exists(sync_path):
                        video_sync_features_path = sync_path
        
        original_duration_from_config = duration
        logging.info(f"[Duration] Config duration: {duration:.2f}s")
        
        accurate_duration, video_fps = get_video_duration_accurate(video_path)
        if accurate_duration is not None and accurate_duration > 0:
            duration = accurate_duration
            logging.info(f"[Duration] Read actual video duration: {duration:.2f}s (replacing config value {original_duration_from_config:.2f}s)")
        else:
            duration = original_duration_from_config if original_duration_from_config and original_duration_from_config > 0 else 10.0
            logging.warning(f"[Duration] Could not read accurate video duration; using config/default value {duration:.2f}s")
        
        logging.info(f"[Duration] Final duration used for generation: {duration:.2f}s")
        
        for idx in range(config.get("each_example_n_times", 1)):
            logging.info(f"--- [Start] Processing seed {seed+idx} ---")
            logging.info(f"Video: {video_path}")
            logging.info(f"Text prompt: {text_prompt}")
            if video_clip_features_path:
                logging.info(f"CLIP features: {video_clip_features_path}")
            if video_sync_features_path:
                logging.info(f"Sync features: {video_sync_features_path}")
            logging.info("Phase 1: Generating audio...")
            
            logging.info(f"[Audio generation] Calling generate_audio_for_video with duration={duration:.2f}s")
            try:
                generated_audio = generate_audio_for_video(
                    foley_omni_engine=foley_omni_engine,
                    video_path=video_path,
                    text_prompt=text_prompt,
                    clip_features_path=video_clip_features_path,
                    sync_features_path=video_sync_features_path,
                    seed=seed+idx,
                    solver_name=solver_name,
                    sample_steps=sample_steps,
                    shift=shift,
                    audio_guidance_scale=audio_guidance_scale,
                    slg_layer=slg_layer,
                    audio_negative_prompt=audio_negative_prompt,
                    duration=duration,
                    cfg_zero_video_features_in_negative=cfg_zero_video_features_in_negative,
                    cfg_apply_to_clip_features=cfg_apply_to_clip_features,
                    cfg_apply_to_sync_features=cfg_apply_to_sync_features,
                    device=device,
                    dtype=target_dtype
                )
            except SkipShortVideoError as e:
                logging.warning(f"[SKIP] {e}")
                torch.cuda.empty_cache()
                continue
            
            torch.cuda.empty_cache()
            
            logging.info("Phase 1 finished. Phase 2: Saving outputs...")
            if sp_rank == 0:
                video_stem = Path(video_path).stem

                output_path = os.path.join(output_dir, f"{video_stem}_{seed+idx}_{global_rank}.mp4")
                audio_output_path = os.path.join(output_dir, f"{video_stem}.wav") if audio_only else output_path.replace('.mp4', '.wav')
                if audio_only:
                    logging.info(f"Saving audio only to: {audio_output_path}")
                    try:
                        import soundfile as sf
                        audio_to_save = generated_audio
                        if audio_to_save.max() > 1.0 or audio_to_save.min() < -1.0:
                            audio_to_save = np.clip(audio_to_save, -1.0, 1.0)
                        sf.write(audio_output_path, audio_to_save, sample_rate)
                        append_jsonl_record(pred_map_jsonl_path, {
                            "video_path": str(video_path),
                            "gt_text": text_prompt,
                            "pred_path": str(audio_output_path),
                            "pred_media_type": "audio",
                            "seed": int(seed + idx),
                            "global_rank": int(global_rank),
                            "sp_rank": int(sp_rank),
                        })
                        logging.info(f"--- [End] Saved successfully: {audio_output_path} ---")
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logging.error(f"Failed to save audio: {e}")
                    continue

                logging.info(f"Saving video to: {output_path}")
                
                try:
                    import av
                    logging.info("Loading source video...")
                    with av.open(video_path) as container:
                        video_stream = container.streams.video[0]
                        
                        video_fps = video_stream.average_rate
                        if video_fps is None:
                            video_fps = video_stream.guessed_rate
                        if video_fps is None:
                            video_fps = 24.0
                        
                        if hasattr(video_fps, 'numerator') and hasattr(video_fps, 'denominator'):
                            video_fps = float(video_fps.numerator) / float(video_fps.denominator)
                        else:
                            video_fps = float(video_fps)
                        
                        logging.info(f"Video frame rate: {video_fps} fps")
                        
                        video_frames = []
                        for frame in container.decode(video_stream):
                            frame_array = frame.to_ndarray(format='rgb24')  # (H, W, C)
                            video_frames.append(frame_array)
                        
                        if not video_frames:
                            raise ValueError("Could not decode video frames")
                        
                        logging.info(f"Decoded {len(video_frames)} video frames")
                        num_frames = len(video_frames)
                        
                        video_duration = num_frames / video_fps
                        logging.info(f"Video duration: {video_duration:.2f}s")

                        target_audio_length = int(video_duration * sample_rate)
                        current_audio_length = len(generated_audio)
                        if current_audio_length != target_audio_length:
                            delta = current_audio_length - target_audio_length
                            delta_sec = abs(delta) / float(sample_rate)
                            logging.info(
                                f"[Audio length adjust: stage 2] before: current={current_audio_length} ({current_audio_length/float(sample_rate):.3f}s), "
                                f"target={target_audio_length} ({target_audio_length/float(sample_rate):.3f}s), "
                                f"delta={delta} samples ({delta_sec:.3f}s)"
                            )

                        if current_audio_length > target_audio_length:
                            generated_audio = generated_audio[:target_audio_length]
                            logging.info("[Audio length adjust: stage 2] Trimmed audio to match video length")
                        elif current_audio_length < target_audio_length:
                            padding = target_audio_length - current_audio_length
                            generated_audio = np.pad(generated_audio, (0, padding), mode='constant')
                            logging.info(f"[Audio length adjust: stage 2] Padded audio by {padding} samples to match video length")

                        if current_audio_length != target_audio_length:
                            logging.info(
                                f"[Audio length adjust: stage 2] after: final={len(generated_audio)} ({len(generated_audio)/float(sample_rate):.3f}s)"
                            )
                        if generated_audio.max() > 1.0 or generated_audio.min() < -1.0:
                            generated_audio = np.clip(generated_audio, -1.0, 1.0)
                        
                        if MMAUDIO_AVAILABLE:
                            fps_frac = Fraction(int(round(video_fps * 1000)), 1000) if video_fps != int(video_fps) else Fraction(int(video_fps), 1)
                            video_info = VideoInfo(
                                duration_sec=video_duration,
                                fps=fps_frac,
                                clip_frames=torch.zeros(1, 1, 3, 224, 224),
                                sync_frames=torch.zeros(1, 1, 3, 224, 224),
                                all_frames=video_frames,
                            )
                            audio_tensor = torch.from_numpy(generated_audio.astype(np.float32)).float().unsqueeze(0)
                            logging.info("Muxing video and audio with PyAV (h264 + aac)...")
                            make_video(video_info, Path(output_path), audio_tensor, sampling_rate=sample_rate)
                        else:
                            logging.info("Stacking video frames with np.stack...")
                            video_array = np.stack(video_frames)
                            del video_frames
                            video_array = np.transpose(video_array, (3, 0, 1, 2))
                            if video_array.max() <= 1.0:
                                video_array = (video_array * 255).astype(np.uint8)
                            else:
                                video_array = video_array.astype(np.uint8)
                            logging.info("Muxing video and audio and encoding MP4 with moviepy...")
                            save_video(output_path, video_array, generated_audio, fps=video_fps, sample_rate=sample_rate)
                            del video_array

                        append_jsonl_record(pred_map_jsonl_path, {
                            "video_path": str(video_path),
                            "gt_text": text_prompt,
                            "pred_path": str(output_path),
                            "pred_media_type": "video",
                            "seed": int(seed + idx),
                            "global_rank": int(global_rank),
                            "sp_rank": int(sp_rank),
                        })
                        
                        logging.info(f"--- [End] Saved successfully: {output_path} ---")
                        torch.cuda.empty_cache()
                        
                except ImportError:
                    logging.error("Missing required libraries (av or moviepy); cannot mux video and audio")
                    import soundfile as sf
                    sf.write(audio_output_path, generated_audio, sample_rate)
                    append_jsonl_record(pred_map_jsonl_path, {
                        "video_path": str(video_path),
                        "gt_text": text_prompt,
                        "pred_path": str(audio_output_path),
                        "pred_media_type": "audio_fallback",
                        "seed": int(seed + idx),
                        "global_rank": int(global_rank),
                        "sp_rank": int(sp_rank),
                    })
                    logging.info(f"Saved audio to: {audio_output_path}")
                except Exception as e:
                    logging.error(f"Error while processing video: {e}")
                    import traceback
                    logging.error(traceback.format_exc())
                    try:
                        import soundfile as sf
                        sf.write(audio_output_path, generated_audio, sample_rate)
                        append_jsonl_record(pred_map_jsonl_path, {
                            "video_path": str(video_path),
                            "gt_text": text_prompt,
                            "pred_path": str(audio_output_path),
                            "pred_media_type": "audio_fallback",
                            "seed": int(seed + idx),
                            "global_rank": int(global_rank),
                            "sp_rank": int(sp_rank),
                        })
                        logging.info(f"Saved audio to: {audio_output_path}")
                    except Exception as e2:
                        logging.error(f"Failed to save fallback audio as well: {e2}")
            
            del generated_audio
            torch.cuda.empty_cache()


def get_arguments(args=sys.argv[1:]):
    parser = get_argument_parser()
    args = parser.parse_args(args)

    # If local_rank wasn't provided, try to infer from common env vars
    if getattr(args, "local_rank", -1) == -1:
        env_lr = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
        try:
            if env_lr is not None:
                args.local_rank = int(env_lr)
        except ValueError:
            pass

    # no cuda mode is not supported
    args.no_cuda = False

    # Optionally bind this process to a specific CUDA device
    if torch.cuda.is_available() and getattr(args, "local_rank", -1) >= 0:
        try:
            torch.cuda.set_device(args.local_rank % torch.cuda.device_count())
        except Exception:
            pass

    return args


def get_argument_parser():
    parser = argparse.ArgumentParser()
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--config-file",
                        type=str,
                        default=os.path.join(_script_dir, "inference_v2st.yaml"))
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    
    return parser

if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config=config, args=args)
