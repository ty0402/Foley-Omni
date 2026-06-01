import logging
import os
import re
import traceback
from pathlib import Path

import torch
from omegaconf import OmegaConf
from optimum.quanto import freeze, qint8, quantize
from tqdm import tqdm

from foley_omni.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from foley_omni.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from foley_omni.utils.model_loading_utils import init_fusion_score_model, init_mmaudio_vae, init_text_model, load_fusion_checkpoint
from diffusers import FlowMatchEulerDiscreteScheduler

_engine_dir = Path(__file__).resolve().parent
_project_root = _engine_dir.parent
_default_config_path = _project_root / "inference_fusion.yaml"
DEFAULT_CONFIG = OmegaConf.load(str(_default_config_path))

NAME_TO_MODEL_SPECS_MAP = {
    "720x720_5s": {
        "path": "model.safetensors",
        "audio_latent_length": 157,
        "formatter": lambda text: re.sub(r"Audio:\s*(.*)", r"<AUDCAP>\1<ENDAUDCAP>", text, flags=re.S),
    },
    "960x960_5s": {
        "path": "model_960x960.safetensors",
        "audio_latent_length": 157,
        "formatter": lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S),
    },
    "960x960_10s": {
        "path": "model_960x960_10s.safetensors",
        "audio_latent_length": 431,
        "formatter": lambda text: re.sub(r"<AUDCAP>(.*?)<ENDAUDCAP>", r"Audio: \1", text, flags=re.S),
    },
}


class FoleyOmniEngine:
    def __init__(self, config=DEFAULT_CONFIG, device=0, target_dtype=torch.bfloat16):
        self.config = config
        self.device = device
        self.target_dtype = target_dtype
        self.cpu_offload = bool(config.get("cpu_offload", False))

        if self.cpu_offload:
            logging.info("CPU offloading is enabled. Initializing non-VAE models on CPU")

        sample_rate = int(config.get("sample_rate", 16000))
        meta_init = True
        model, _, audio_config = init_fusion_score_model(rank=device, meta_init=meta_init, sample_rate=sample_rate)

        fp8 = bool(config.get("fp8", False))
        int8 = bool(config.get("qint8", False))

        if not meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()

        self.vae_model_audio = init_mmaudio_vae(config.ckpt_dir, rank=device, sample_rate=sample_rate)
        self.vae_model_audio.requires_grad_(False).eval()
        self.vae_model_audio = self.vae_model_audio.bfloat16()

        self.text_model = init_text_model(config.ckpt_dir, rank=device, cpu_offload=self.cpu_offload)
        if config.get("shard_text_model", False):
            raise NotImplementedError("Sharding text model is not implemented in the public release.")
        if self.cpu_offload:
            self.offload_to_cpu(self.text_model.model)

        model_name = config.get("model_name", "960x960_10s")
        if model_name not in NAME_TO_MODEL_SPECS_MAP:
            raise ValueError(f"Unknown model_name: {model_name}")
        self.model_name = model_name
        model_specs = NAME_TO_MODEL_SPECS_MAP[model_name]

        model_checkpoint = config.get("model_checkpoint")
        if model_checkpoint is not None:
            checkpoint_path = model_checkpoint
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"Required model checkpoint not found: {checkpoint_path}")
            logging.info(f"Loading model checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict, strict=False, assign=True)
        else:
            basename = model_specs["path"]
            if fp8:
                if model_name != "720x720_5s":
                    raise ValueError("FP8 quantization is only supported for the 720x720_5s checkpoint.")
                basename = "model_fp8_e4m3fn.safetensors"
            checkpoint_path = os.path.join(config.ckpt_dir, "fusion", basename)
            if not os.path.exists(checkpoint_path):
                raise RuntimeError(f"Required checkpoint not found: {checkpoint_path}")
            load_fusion_checkpoint(model, checkpoint_path=checkpoint_path, from_meta=meta_init)

        if meta_init:
            if not fp8:
                model = model.to(dtype=target_dtype)
            model = model.to(device=device if not self.cpu_offload else "cpu").eval()
            model.set_rope_params()

        self.model = model
        if int8:
            quantize(self.model, qint8)
            freeze(self.model)

        self.audio_latent_channel = audio_config.get("in_dim")
        self.audio_latent_length = model_specs["audio_latent_length"]
        if self.model_name == "960x960_10s" and sample_rate == 16000:
            self.audio_latent_length = 314
        self.text_formatter = model_specs["formatter"]

        logging.info(
            "FoleyOmni audio-only engine initialized, cpu_offload=%s. GPU VRAM allocated: %.2f GB, reserved: %.2f GB",
            self.cpu_offload,
            torch.cuda.memory_allocated(device) / 1e9,
            torch.cuda.memory_reserved(device) / 1e9,
        )

    @torch.inference_mode()
    def generate(
        self,
        text_prompt,
        seed=100,
        solver_name="unipc",
        sample_steps=50,
        shift=5.0,
        audio_guidance_scale=4.0,
        slg_layer=9,
        audio_negative_prompt="",
    ):
        params = {
            "Text Prompt": text_prompt,
            "Seed": seed,
            "Solver": solver_name,
            "Sample Steps": sample_steps,
            "Shift": shift,
            "Audio Guidance Scale": audio_guidance_scale,
            "SLG Layer": slg_layer,
            "Audio Negative Prompt": audio_negative_prompt,
        }
        pretty = "\n".join(f"{k:>24}: {v}" for k, v in params.items())
        logging.info("\n========== Generation Parameters ==========\n%s\n==========================================", pretty)

        try:
            scheduler_audio, timesteps_audio = self.get_scheduler_time_steps(
                sampling_steps=sample_steps,
                device=self.device,
                solver_name=solver_name,
                shift=shift,
            )

            formatted_text_prompt = self.text_formatter(text_prompt)
            if formatted_text_prompt != text_prompt:
                logging.info(
                    "Prompt format was normalized for checkpoint compatibility. Original prompt: %s | Formatted prompt: %s",
                    text_prompt,
                    formatted_text_prompt,
                )
                text_prompt = formatted_text_prompt

            if self.cpu_offload:
                self.text_model.model = self.text_model.model.to(self.device)
            text_embeddings = self.text_model([text_prompt, audio_negative_prompt], self.text_model.device)
            text_embeddings = [emb.to(self.target_dtype).to(self.device) for emb in text_embeddings]
            if self.cpu_offload:
                self.offload_to_cpu(self.text_model.model)

            text_embeddings_audio_pos = text_embeddings[0]
            text_embeddings_audio_neg = text_embeddings[1]

            audio_noise = torch.randn(
                (self.audio_latent_length, self.audio_latent_channel),
                device=self.device,
                dtype=self.target_dtype,
                generator=torch.Generator(device=self.device).manual_seed(seed),
            )
            max_seq_len_audio = audio_noise.shape[0]

            if self.cpu_offload:
                self.offload_to_cpu(self.vae_model_audio)
                self.model = self.model.to(self.device)

            with torch.amp.autocast("cuda", enabled=self.target_dtype != torch.float32, dtype=self.target_dtype):
                for _, t_a in tqdm(enumerate(timesteps_audio), total=len(timesteps_audio), desc="Sampling"):
                    timestep_input = torch.full((1,), t_a, device=self.device)

                    pos_forward_args = {
                        "audio_context": [text_embeddings_audio_pos],
                        "vid_context": None,
                        "vid_seq_len": None,
                        "audio_seq_len": max_seq_len_audio,
                        "first_frame_is_clean": False,
                    }
                    _, pred_audio_pos = self.model(
                        vid=None,
                        audio=[audio_noise],
                        t=timestep_input,
                        **pos_forward_args,
                    )

                    neg_forward_args = {
                        "audio_context": [text_embeddings_audio_neg],
                        "vid_context": None,
                        "vid_seq_len": None,
                        "audio_seq_len": max_seq_len_audio,
                        "first_frame_is_clean": False,
                        "slg_layer": slg_layer,
                    }
                    _, pred_audio_neg = self.model(
                        vid=None,
                        audio=[audio_noise],
                        t=timestep_input,
                        **neg_forward_args,
                    )

                    pred_audio_guided = pred_audio_neg[0] + audio_guidance_scale * (pred_audio_pos[0] - pred_audio_neg[0])
                    audio_noise = scheduler_audio.step(
                        pred_audio_guided.unsqueeze(0),
                        t_a,
                        audio_noise.unsqueeze(0),
                        return_dict=False,
                    )[0].squeeze(0)

                if self.cpu_offload:
                    self.offload_to_cpu(self.model)
                    self.vae_model_audio = self.vae_model_audio.to(self.device)

                audio_latents_for_vae = audio_noise.unsqueeze(0).transpose(1, 2)
                generated_audio = self.vae_model_audio.wrapped_decode(audio_latents_for_vae)
                generated_audio = generated_audio.squeeze().cpu().float().numpy()

                if self.cpu_offload:
                    self.offload_to_cpu(self.vae_model_audio)

            return None, generated_audio, None

        except Exception:
            logging.error(traceback.format_exc())
            raise

    def offload_to_cpu(self, model):
        model = model.cpu()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return model

    def get_scheduler_time_steps(self, sampling_steps, solver_name="unipc", device=0, shift=5.0):
        torch.manual_seed(4)

        if solver_name == "unipc":
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False,
            )
            sample_scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
            timesteps = sample_scheduler.timesteps
        elif solver_name == "dpm++":
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=1000,
                shift=1,
                use_dynamic_shifting=False,
            )
            sampling_sigmas = get_sampling_sigmas(sampling_steps, shift=shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, device=device, sigmas=sampling_sigmas)
        elif solver_name == "euler":
            sample_scheduler = FlowMatchEulerDiscreteScheduler(shift=shift)
            timesteps, _ = retrieve_timesteps(sample_scheduler, sampling_steps, device=device)
        else:
            raise NotImplementedError("Unsupported solver.")

        return sample_scheduler, timesteps
