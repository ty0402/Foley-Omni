import torch
import torch.nn as nn
from wan.modules.model import WanLayerNorm, WanModel, WanRMSNorm, rope_apply
from wan.modules.attention import flash_attention

class MusicModel(nn.Module):
    def __init__(self, audio_config=None):
        super().__init__()

        self.audio_model = WanModel(**audio_config)

        self.num_blocks = len(self.audio_model.blocks)

        self.init_weights()


    def forward(
        self,
        audio,
        t,
        audio_context,
        audio_seq_len,
        clip_fea_audio=None,
        clip_features=None,
        sync_features=None,
        y=None,
    ):  

        return self.audio_model(
            x=audio, 
            t=t, 
            context=audio_context, 
            seq_len=audio_seq_len, 
            clip_fea=clip_fea_audio,
            clip_features=clip_features,
            sync_features=sync_features,
            y=None
        )    

    def init_weights(self):
        if self.audio_model is not None:
            self.audio_model.init_weights()

    
    def set_rope_params(self):
        self.audio_model.set_rope_params()


    def enable_gradient_checkpointing(self):
        """
        Enables gradient checkpointing for both video and audio models.
        """
        print("Enabling gradient checkpointing for MusicModel...")
        if hasattr(self, 'audio_model'):
            self.audio_model.enable_gradient_checkpointing()