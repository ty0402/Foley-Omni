#!/usr/bin/env python3
"""Extract CLIP and Synchformer features from MP4 videos and save them as .npy files."""

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "mmaudio"))

from mmaudio.eval_utils import load_video
from mmaudio.model.utils.features_utils import FeaturesUtils

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s")

DEFAULT_DURATION_SEC = float("inf")
MODE = "16k"


def resolve_first_existing_file(filename: str, candidates: list[Path]) -> Path:
    for base in candidates:
        path = base / filename
        if path.exists():
            return path
    return candidates[0] / filename


def get_ext_weights_candidates() -> list[Path]:
    candidates = []
    env_root = os.environ.get("VRFOLEY_EXT_WEIGHTS")
    if env_root:
        candidates.append(Path(env_root))
    candidates.extend([
        REPO_ROOT / "ckpts" / "mmaudio" / "ext_weights",
        Path("/taoye/workspace/VRSound/ext_weights"),
        Path("/taoye/ty/data/ckpt/MMAudio/ext_weights"),
        Path("./ext_weights"),
    ])
    return candidates


EXT_WEIGHTS_CANDIDATES = get_ext_weights_candidates()
VAE_CKPT = resolve_first_existing_file("v1-16.pth", EXT_WEIGHTS_CANDIDATES)
VOCODER_CKPT = resolve_first_existing_file("best_netG.pt", EXT_WEIGHTS_CANDIDATES)
SYNCHFORMER_CKPT = resolve_first_existing_file("synchformer_state_dict.pth", EXT_WEIGHTS_CANDIDATES)


def check_required_files() -> None:
    for path in [VAE_CKPT, VOCODER_CKPT, SYNCHFORMER_CKPT]:
        if not path.exists():
            raise FileNotFoundError(f"Required feature extraction dependency is missing: {path}")


def process_videos_on_gpu(
    video_files: list[str],
    feature_dir: str,
    gpu_id: int,
    vae_path: str,
    bigvgan_path: str,
    synchformer_ckpt: str,
    mode: str,
    duration_sec: float,
    progress_queue,
):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")

    feature_extractor = FeaturesUtils(
        tod_vae_ckpt=None,
        enable_conditions=True,
        bigvgan_vocoder_ckpt=None,
        synchformer_ckpt=synchformer_ckpt,
        mode=mode,
        need_vae_encoder=False,
    ).eval().to(device)

    results = []
    for video_path_str in video_files:
        try:
            video_path = Path(video_path_str)
            video_stem = video_path.stem
            parent_name = video_path.parent.name
            feature_stem = f"{parent_name}_{video_stem}" if parent_name else video_stem

            video_info = load_video(video_path, duration_sec)
            clip_frames = video_info.clip_frames.unsqueeze(0).to(device)
            sync_frames = video_info.sync_frames.unsqueeze(0).to(device)

            with torch.no_grad():
                clip_features = feature_extractor.encode_video_with_clip(clip_frames)
                sync_features = feature_extractor.encode_video_with_sync(sync_frames)

            clip_features_np = clip_features.squeeze(0).detach().cpu().float().numpy()
            sync_features_np = sync_features.squeeze(0).detach().cpu().float().numpy()

            feature_dir_path = Path(feature_dir)
            clip_output_path = feature_dir_path / f"{feature_stem}_clip_features.npy"
            sync_output_path = feature_dir_path / f"{feature_stem}_sync_features.npy"

            np.save(clip_output_path, clip_features_np)
            np.save(sync_output_path, sync_features_np)

            results.append({
                "audio_path": str(video_path),
                "clip_feature_path": str(clip_output_path.absolute()),
                "sync_feature_path": str(sync_output_path.absolute()),
            })
        except Exception as exc:
            log.error("Error processing %s on GPU %s: %s", video_path_str, gpu_id, exc)
        finally:
            progress_queue.put(1)

    return results


def extract_features_from_videos(
    *,
    video_dir: str | None,
    feature_dir: str,
    json_output: str,
    json_input: str | None,
    num_gpus: int | None,
    gpu_ids: list[int] | None,
    skip_existing: bool,
    workers_per_gpu: int,
):
    feature_dir_path = Path(feature_dir)
    json_output_path = Path(json_output)
    feature_dir_path.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    video_files = []
    existing_json_data = {}
    if json_input and Path(json_input).exists():
        log.info("Reading video list from JSON file: %s", json_input)
        with open(json_input, "r", encoding="utf-8") as f:
            existing_json_data = json.load(f)

        for video_path, video_info in existing_json_data.items():
            video_path_obj = Path(video_path)
            if not video_path_obj.exists():
                log.warning("Video file not found, skipping: %s", video_path)
                continue

            if skip_existing:
                clip_feature_path = video_info.get("clip_feature_path")
                sync_feature_path = video_info.get("sync_feature_path")
                if clip_feature_path and sync_feature_path:
                    if Path(clip_feature_path).exists() and Path(sync_feature_path).exists():
                        continue

            video_files.append(video_path_obj)

        log.info(
            "Found %s videos to process from JSON (skipped %s with existing features)",
            len(video_files),
            len(existing_json_data) - len(video_files),
        )
    else:
        if not video_dir:
            raise ValueError("Either --video_dir or --json_input must be provided")

        video_dir_path = Path(video_dir)
        if not video_dir_path.exists():
            raise FileNotFoundError(f"Video directory not found: {video_dir_path}")

        video_files = list(video_dir_path.glob("*.mp4"))
        if not video_files:
            log.warning("No MP4 files found in %s", video_dir_path)
            return

        log.info("Found %s video files in directory", len(video_files))

    if not video_files:
        log.warning("No videos to process")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. This script requires GPU support.")

    available_gpus = torch.cuda.device_count()
    if gpu_ids is not None:
        for gpu_id in gpu_ids:
            if gpu_id < 0 or gpu_id >= available_gpus:
                raise ValueError(f"GPU ID {gpu_id} is invalid. Available GPUs: 0-{available_gpus - 1}")
        gpu_id_list = gpu_ids
        num_gpus = len(gpu_id_list)
        log.info("Using specified GPU IDs: %s", gpu_id_list)
    elif num_gpus is not None:
        num_gpus = min(num_gpus, available_gpus)
        gpu_id_list = list(range(num_gpus))
        log.info("Using first %s GPUs: %s", num_gpus, gpu_id_list)
    else:
        num_gpus = available_gpus
        gpu_id_list = list(range(available_gpus))
        log.info("Using all available GPUs: %s", gpu_id_list)

    total_workers = num_gpus * workers_per_gpu
    log.info(
        "Available GPUs: %s, using %s GPUs: %s, %s workers/GPU -> %s workers",
        available_gpus,
        num_gpus,
        gpu_id_list,
        workers_per_gpu,
        total_workers,
    )

    video_files_str = [str(v) for v in video_files]
    videos_per_worker = max(1, len(video_files_str) // total_workers)
    video_chunks = []
    for i in range(total_workers):
        start_idx = i * videos_per_worker
        if i == total_workers - 1:
            end_idx = len(video_files_str)
        else:
            end_idx = min(len(video_files_str), (i + 1) * videos_per_worker)
        chunk = video_files_str[start_idx:end_idx]
        gpu_id = gpu_id_list[i // workers_per_gpu]
        video_chunks.append((chunk, gpu_id))

    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    manager = mp.Manager()
    progress_queue = manager.Queue()
    process_args = []
    for chunk, gpu_id in video_chunks:
        if chunk:
            process_args.append((
                chunk,
                str(feature_dir_path),
                gpu_id,
                str(VAE_CKPT),
                str(VOCODER_CKPT),
                str(SYNCHFORMER_CKPT),
                MODE,
                DEFAULT_DURATION_SEC,
                progress_queue,
            ))

    new_json_data = {}
    with mp.Pool(processes=total_workers) as pool:
        processes = [pool.apply_async(process_videos_on_gpu, args) for args in process_args]

        completed = 0
        with tqdm(total=len(video_files), desc="Extracting features") as pbar:
            while completed < len(video_files):
                try:
                    progress_queue.get(timeout=1)
                    completed += 1
                    pbar.update(1)
                except Exception:
                    if all(p.ready() for p in processes):
                        break

        for process in processes:
            try:
                results = process.get(timeout=3600)
                for result in results:
                    audio_path = result["audio_path"]
                    new_json_data[audio_path] = {
                        "clip_feature_path": result["clip_feature_path"],
                        "sync_feature_path": result["sync_feature_path"],
                    }
            except Exception as exc:
                log.error("Error collecting worker result: %s", exc)

    final_json_data = existing_json_data.copy() if existing_json_data else {}
    for audio_path, feature_data in new_json_data.items():
        if audio_path in final_json_data:
            final_json_data[audio_path].update(feature_data)
        else:
            final_json_data[audio_path] = {"resp": "", **feature_data}

    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(final_json_data, f, indent=2, ensure_ascii=False)

    log.info("Extraction completed.")
    log.info("Processed %s/%s videos", len(new_json_data), len(video_files))
    log.info("Features saved to: %s", feature_dir_path)
    log.info("JSON metadata saved to: %s", json_output_path)
    log.info("Using VAE: %s", VAE_CKPT)
    log.info("Using vocoder: %s", VOCODER_CKPT)
    log.info("Using Synchformer: %s", SYNCHFORMER_CKPT)


def parse_gpu_ids(raw_gpu_ids: str | None) -> list[int] | None:
    if raw_gpu_ids is None:
        return None
    try:
        gpu_ids = [int(x.strip()) for x in raw_gpu_ids.split(",") if x.strip()]
    except ValueError as exc:
        raise ValueError(
            f"Invalid --gpu_ids format: {raw_gpu_ids}. Expected comma-separated integers such as '0,1,2'."
        ) from exc
    if not gpu_ids:
        raise ValueError("--gpu_ids cannot be empty")
    return gpu_ids


def main():
    parser = argparse.ArgumentParser(
        description="Extract CLIP and Synchformer features from MP4 videos and generate JSON metadata."
    )
    parser.add_argument("--video_dir", type=str, default=None, help="Directory containing MP4 videos.")
    parser.add_argument("--json_input", type=str, default=None, help="Optional input JSON manifest.")
    parser.add_argument("--feature_dir", type=str, required=True, help="Output directory for .npy feature files.")
    parser.add_argument("--json_output", type=str, required=True, help="Output JSON manifest path.")
    parser.add_argument(
        "--skip_existing",
        action="store_true",
        default=True,
        help="Skip videos that already have extracted features.",
    )
    parser.add_argument(
        "--no_skip_existing",
        action="store_false",
        dest="skip_existing",
        help="Process all videos even if features already exist.",
    )
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use.")
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use, such as '0,1,2'. Takes priority over --num_gpus.",
    )
    parser.add_argument(
        "--workers_per_gpu",
        type=int,
        default=4,
        help="Number of worker processes to launch per GPU.",
    )
    args = parser.parse_args()

    check_required_files()
    gpu_ids = parse_gpu_ids(args.gpu_ids)
    extract_features_from_videos(
        video_dir=args.video_dir,
        feature_dir=args.feature_dir,
        json_output=args.json_output,
        json_input=args.json_input,
        num_gpus=args.num_gpus,
        gpu_ids=gpu_ids,
        skip_existing=args.skip_existing,
        workers_per_gpu=args.workers_per_gpu,
    )


if __name__ == "__main__":
    main()
