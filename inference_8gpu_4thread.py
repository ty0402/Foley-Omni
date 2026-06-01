import argparse
import copy
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from omegaconf import OmegaConf


DEFAULT_CKPT_DIR = str(Path(__file__).parent / "checkpoints")


def extract_epoch_from_filename(filename: str) -> Optional[int]:
    """Extract the epoch number from a checkpoint filename."""
    match = re.search(r'(?:model-)?epoch-(\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    match = re.search(r'epoch[_-](\d+)', filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def collect_checkpoints(ckpt_dir: Path, epochs: Optional[List[int]] = None) -> List[Path]:
    """Collect `.pth` checkpoints and keep them sorted by filename."""
    if not ckpt_dir.is_dir():
        return []
    files = sorted(ckpt_dir.glob("*.pth"))
    
    if epochs is not None:
        filtered = []
        for f in files:
            epoch = extract_epoch_from_filename(f.name)
            if epoch is not None and epoch in epochs:
                filtered.append(f)
            elif epoch is None:
                print(f"[WARN] Could not extract epoch from filename; skipping: {f.name}")
        return filtered
    
    return files


def parse_gpu_ids(gpu_ids: str) -> List[int]:
    values = [x.strip() for x in gpu_ids.split(",") if x.strip()]
    if not values:
        raise ValueError("`--gpu-ids` is empty.")
    parsed = [int(x) for x in values]
    if len(set(parsed)) != len(parsed):
        raise ValueError("`--gpu-ids` contains duplicate ids.")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch inference_v2st.py with 8 GPUs and 4 threads per GPU process."
    )
    parser.add_argument(
        "--base-config",
        type=str,
        default=str(Path(__file__).parent / "inference_v2st.yaml"),
        help="Base config path.",
    )
    parser.add_argument(
        "--inference-script",
        type=str,
        default=str(Path(__file__).parent / "inference_v2st.py"),
        help="Inference script path.",
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default="0,1,2,3,4,5,6,7",
        help="Visible physical GPU ids.",
    )
    parser.add_argument(
        "--procs-per-gpu",
        type=int,
        default=2,
        help="Number of Python processes to launch per physical GPU.",
    )
    parser.add_argument(
        "--threads-per-gpu",
        type=int,
        default=4,
        help="OMP/MKL thread count per GPU process.",
    )
    parser.add_argument(
        "--sp-size",
        type=int,
        default=None,
        help="Optional override for sp_size in runtime config.",
    )
    parser.add_argument(
        "--work-dir",
        type=str,
        default=str(Path(__file__).parent / "output" / "multi_infer_8gpu_4thread"),
        help="Directory for runtime config and log.",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=DEFAULT_CKPT_DIR,
        help="Directory containing .pth checkpoints to test (all will be run).",
    )
    parser.add_argument(
        "--epochs",
        type=str,
        default=None,
        help="Comma-separated list of epoch numbers to test (e.g., '39,40,41'). If not specified, all checkpoints will be tested.",
    )
    parser.add_argument(
        "--master-port",
        type=int,
        default=29500,
        help="Torch distributed master port.",
    )
    args = parser.parse_args()

    base_config_path = Path(args.base_config).resolve()
    inference_script_path = Path(args.inference_script).resolve()
    work_dir = Path(args.work_dir).resolve()
    ckpt_dir = Path(args.ckpt_dir).resolve()

    if not base_config_path.is_file():
        raise FileNotFoundError(f"Base config not found: {base_config_path}")
    if not inference_script_path.is_file():
        raise FileNotFoundError(f"Inference script not found: {inference_script_path}")
    if args.threads_per_gpu <= 0:
        raise ValueError("`--threads-per-gpu` must be positive.")
    if args.procs_per_gpu <= 0:
        raise ValueError("`--procs-per-gpu` must be positive.")

    epochs = None
    if args.epochs:
        try:
            epochs = [int(x.strip()) for x in args.epochs.split(",") if x.strip()]
            if not epochs:
                raise ValueError("`--epochs` cannot be empty")
            print(f"[INFO] Will test only these epochs: {epochs}")
        except ValueError as e:
            raise ValueError(f"Invalid `--epochs` format; expected comma-separated integers such as '39,40,41': {e}")

    checkpoints = collect_checkpoints(ckpt_dir, epochs=epochs)
    if not checkpoints:
        if epochs:
            raise FileNotFoundError(
                f"No .pth files matching epochs {epochs} were found under checkpoint directory: {ckpt_dir}"
            )
        else:
            raise FileNotFoundError(
                f"No .pth files were found under checkpoint directory: {ckpt_dir}"
            )
    
    if epochs:
        print(f"[INFO] Found {len(checkpoints)} matching checkpoints; they will be tested in order.")
        for ckpt in checkpoints:
            epoch = extract_epoch_from_filename(ckpt.name)
            print(f"  - {ckpt.name} (epoch {epoch})")
    else:
        print(f"[INFO] Found {len(checkpoints)} checkpoints; they will be tested in order.")

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    num_gpus = len(gpu_ids)
    world_size = num_gpus * args.procs_per_gpu
    if num_gpus != 8:
        print(f"[WARN] Detected {num_gpus} GPUs (expected 8). Launching with the available GPU count and {args.procs_per_gpu} processes per GPU for a total world size of {world_size}.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = work_dir / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = OmegaConf.load(str(base_config_path))
    base_out = Path(str(base_cfg.get("output_dir", "./outputs")))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(x) for x in gpu_ids)
    env["OMP_NUM_THREADS"] = str(args.threads_per_gpu)
    env["MKL_NUM_THREADS"] = str(args.threads_per_gpu)

    for idx, ckpt_path in enumerate(checkpoints):
        ckpt_stem = ckpt_path.stem
        output_dir = base_out / ckpt_stem
        if output_dir.exists():
            print(f"\n[SKIP] [{idx + 1}/{len(checkpoints)}] Output already exists; skipping: {ckpt_stem} -> {output_dir}")
            continue
        print(f"\n[INFO] [{idx + 1}/{len(checkpoints)}] Testing checkpoint: {ckpt_path}")

        runtime_cfg = copy.deepcopy(base_cfg)
        runtime_cfg["model_checkpoint"] = str(ckpt_path.resolve())
        if args.sp_size is not None:
            runtime_cfg["sp_size"] = int(args.sp_size)
        runtime_cfg["audio_only"] = True
        runtime_cfg["output_dir"] = str(output_dir)

        runtime_cfg_path = run_dir / f"inference_runtime_{ckpt_stem}.yaml"
        OmegaConf.save(config=runtime_cfg, f=str(runtime_cfg_path))

        cmd = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nnodes=1",
            "--nproc_per_node",
            str(world_size),
            "--master_port",
            str(args.master_port),
            str(inference_script_path),
            "--config-file",
            str(runtime_cfg_path),
        ]

        log_path = run_dir / f"run_{ckpt_stem}.log"
        print(f"[INFO] runtime config: {runtime_cfg_path}")
        print(f"[INFO] output_dir: {runtime_cfg['output_dir']}")
        print(f"[INFO] log path: {log_path}")
        print(f"[INFO] CMD: {' '.join(cmd)}")

        with log_path.open("w", encoding="utf-8") as f:
            f.write(f"checkpoint={ckpt_path}\n")
            f.write(f"CUDA_VISIBLE_DEVICES={env['CUDA_VISIBLE_DEVICES']}\n")
            f.write(f"OMP_NUM_THREADS={env['OMP_NUM_THREADS']}\n")
            f.write(f"MKL_NUM_THREADS={env['MKL_NUM_THREADS']}\n")
            f.write(f"CMD={' '.join(cmd)}\n\n")
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=str(inference_script_path.parent),
                bufsize=1,
                universal_newlines=True,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                f.flush()
                sys.stdout.write(line)
                sys.stdout.flush()
            ret = proc.wait()

        if ret != 0:
            raise SystemExit(
                f"[ERR] checkpoint {ckpt_stem} inference failed, exit_code={ret}, log={log_path}"
            )
        print(f"[DONE] checkpoint {ckpt_stem} finished. Log: {log_path}")

    print(f"\n[DONE] Finished testing all {len(checkpoints)} checkpoints. run_dir: {run_dir}")


if __name__ == "__main__":
    main()
