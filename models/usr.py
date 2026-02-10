import torch
import torch.nn as nn
from hydra.utils import instantiate


class USRModel(nn.Module):
    def __init__(self, cfg, backbone_args=None, pred_other=None):
        super().__init__()
        self.cfg = cfg
        self.odim = 1049
        self.ignore_id = -1
        self.backbone = instantiate(cfg.model.backbone)
        self.sos = self.odim - 1
        self.eos = self.odim - 1

    def get_encoded_features(self, video, audio, padding_mask):
        return self.backbone.encoder(
            xs_v=video, xs_a=audio, masks=padding_mask, return_all=True,
        )


class USR(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.model = USRModel(cfg)
        self.cfg = cfg
