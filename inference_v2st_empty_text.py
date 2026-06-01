import os
from omegaconf import OmegaConf

from inference_v2st import main, get_arguments


def load_config_and_force_empty_text(config_path: str):

    config = OmegaConf.load(config_path)

    config["text_prompt"] = ""

    config["force_empty_text"] = True

    if "audio_negative_prompt" in config:
        config["audio_negative_prompt"] = ""

    return config


if __name__ == "__main__":
    args = get_arguments()

    if not os.path.isfile(args.config_file):
        raise FileNotFoundError(f"Config file not found: {args.config_file}")

    config = load_config_and_force_empty_text(args.config_file)
    main(config=config, args=args)
