import argparse
import json
import logging
import os
import sys

import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from foley_omni.distributed_comms.parallel_states import get_sequence_parallel_state, initialize_sequence_parallel_state, nccl_info
from foley_omni.distributed_comms.util import get_global_rank, get_local_rank, get_world_size
from foley_omni.fusion_engine import FoleyOmniEngine
from foley_omni.utils.processing_utils import format_prompt_for_filename, validate_and_process_user_prompt


def _init_logging(rank):
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s] %(levelname)s: %(message)s",
            handlers=[logging.StreamHandler(stream=sys.stdout)],
        )
    else:
        logging.basicConfig(level=logging.ERROR)


def main(config, args):
    world_size = get_world_size()
    global_rank = get_global_rank()
    local_rank = get_local_rank()
    device = local_rank
    torch.cuda.set_device(local_rank)

    sp_size = config.get("sp_size", 1)
    assert sp_size <= world_size and world_size % sp_size == 0, "sp_size must divide world_size"

    _init_logging(global_rank)

    if world_size > 1:
        torch.distributed.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=global_rank,
            world_size=world_size,
        )
    else:
        assert sp_size == 1, f"When world_size is 1, sp_size must also be 1, but got {sp_size}."

    initialize_sequence_parallel_state(sp_size)
    logging.info("Using SP: %s, SP_SIZE: %s", get_sequence_parallel_state(), sp_size)

    args.local_rank = local_rank
    args.device = device
    target_dtype = torch.bfloat16

    text_prompt = config.get("text_prompt")
    mode = config.get("mode", "vt2a")
    assert mode in {"t2a", "vt2a"}, f"Invalid mode {mode}; the public text-only entrypoint only supports t2a/vt2a"

    text_prompts, _, output_stems, source_row_ids = validate_and_process_user_prompt(
        text_prompt,
        image_path=None,
        mode="vt2a",
    )

    logging.info("Loading FoleyOmni engine...")
    foley_omni_engine = FoleyOmniEngine(config=config, device=device, target_dtype=target_dtype)
    logging.info("FoleyOmni engine loaded")

    output_dir = config.get("output_dir", "./outputs")
    os.makedirs(output_dir, exist_ok=True)
    mapping_filename = config.get("mapping_filename", "inference_path_text_map.json")
    mapping_path = os.path.join(output_dir, mapping_filename)

    all_eval_data = list(zip(text_prompts, output_stems, source_row_ids))

    use_sp = get_sequence_parallel_state()
    if use_sp:
        sp_size = nccl_info.sp_size
        sp_rank = nccl_info.rank_within_group
        sp_group_id = global_rank // sp_size
        num_sp_groups = world_size // sp_size
    else:
        sp_rank = 0
        sp_group_id = global_rank
        num_sp_groups = world_size

    total_files = len(all_eval_data)
    if total_files == 0:
        logging.error("ERROR: No evaluation files found")
        this_rank_eval_data = []
    else:
        this_rank_eval_data = all_eval_data[sp_group_id::num_sp_groups]

    local_mapping_records = []
    sample_rate = int(config.get("sample_rate", 16000))
    solver_name = config.get("solver_name", "unipc")
    sample_steps = config.get("sample_steps", 50)
    shift = config.get("shift", 5.0)
    audio_guidance_scale = config.get("audio_guidance_scale", 3.0)
    slg_layer = config.get("slg_layer", 11)
    audio_negative_prompt = config.get("audio_negative_prompt", "")
    seed = config.get("seed", 100)

    for _, (text_prompt_item, output_stem, source_row_id) in tqdm(enumerate(this_rank_eval_data)):
        for idx in range(config.get("each_example_n_times", 1)):
            current_seed = seed + idx
            logging.info("--- [Start] Processing seed %s ---", current_seed)
            logging.info("Phase 1: Diffusion sampling and VAE decoding...")

            _, generated_audio, _ = foley_omni_engine.generate(
                text_prompt=text_prompt_item,
                seed=current_seed,
                solver_name=solver_name,
                sample_steps=sample_steps,
                shift=shift,
                audio_guidance_scale=audio_guidance_scale,
                slg_layer=slg_layer,
                audio_negative_prompt=audio_negative_prompt,
            )

            logging.info("Phase 1 finished. Phase 2: Saving output...")
            if sp_rank == 0:
                if output_stem is not None:
                    base_name = output_stem
                    use_id_naming = True
                else:
                    base_name = format_prompt_for_filename(text_prompt_item)
                    use_id_naming = False

                rank_suffix = "" if use_id_naming else f"_{global_rank}"
                output_path = os.path.join(output_dir, f"{base_name}_{current_seed}{rank_suffix}.wav")

                import soundfile as sf

                logging.info("Saving audio to: %s", output_path)
                audio_data = generated_audio.detach().cpu().numpy() if isinstance(generated_audio, torch.Tensor) else generated_audio
                sf.write(output_path, audio_data, sample_rate)
                logging.info("--- [End] Audio saved successfully: %s ---", output_path)

                rec = {
                    "output_path": output_path,
                    "text_prompt": text_prompt_item,
                    "output_stem": base_name,
                }
                if source_row_id is not None:
                    rec["id"] = source_row_id
                local_mapping_records.append(rec)

    all_mapping_records = local_mapping_records
    if world_size > 1 and torch.distributed.is_initialized():
        gathered_records = [None for _ in range(world_size)]
        torch.distributed.all_gather_object(gathered_records, local_mapping_records)
        if global_rank == 0:
            all_mapping_records = []
            for rank_records in gathered_records:
                if rank_records:
                    all_mapping_records.extend(rank_records)

    if global_rank == 0:
        with open(mapping_path, "w", encoding="utf-8") as f:
            json.dump(all_mapping_records, f, ensure_ascii=False, indent=2)
        logging.info("Mapping file saved to: %s (rows=%s)", mapping_path, len(all_mapping_records))


def get_arguments(args=sys.argv[1:]):
    parser = get_argument_parser()
    args = parser.parse_args(args)

    if getattr(args, "local_rank", -1) == -1:
        env_lr = os.environ.get("LOCAL_RANK") or os.environ.get("SLURM_LOCALID")
        try:
            if env_lr is not None:
                args.local_rank = int(env_lr)
        except ValueError:
            pass

    args.no_cuda = False
    if torch.cuda.is_available() and getattr(args, "local_rank", -1) >= 0:
        try:
            torch.cuda.set_device(args.local_rank % torch.cuda.device_count())
        except Exception:
            pass

    return args


def get_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="inference_fusion.yaml")
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for distributed inference")
    return parser


if __name__ == "__main__":
    args = get_arguments()
    config = OmegaConf.load(args.config_file)
    main(config=config, args=args)
