import torch.nn as nn

from foley_omni.distributed_comms.parallel_states import get_sequence_parallel_state
from foley_omni.modules.model import WanModel


class FusionModel(nn.Module):
    """Public audio-only wrapper around the released Foley-Omni audio tower."""

    def __init__(self, video_config=None, audio_config=None):
        super().__init__()
        if audio_config is None:
            raise ValueError("audio_config must be provided for the public audio-only release")

        if video_config is not None:
            print("Warning: video_config is ignored in the public audio-only release.")

        self.video_model = None
        self.audio_model = WanModel(**audio_config)
        self.num_blocks = len(self.audio_model.blocks)
        self.use_sp = get_sequence_parallel_state()

    def forward(
        self,
        vid,
        audio,
        t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        clip_features=None,
        sync_features=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
    ):
        del vid, vid_context, vid_seq_len, clip_fea, y, first_frame_is_clean, slg_layer

        if audio is None or all(x is None for x in audio):
            raise ValueError("Audio latents must be provided for the public audio-only release")

        audio_output = self.audio_model(
            x=audio,
            t=t,
            context=audio_context,
            seq_len=audio_seq_len,
            clip_fea=clip_fea_audio,
            clip_features=clip_features,
            sync_features=sync_features,
            y=None,
        )
        return None, audio_output

    def init_weights(self):
        self.audio_model.init_weights()

    def set_rope_params(self):
        self.audio_model.set_rope_params()
