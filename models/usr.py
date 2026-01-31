import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E


class USRModel(nn.Module):
    def __init__(self, cfg, backbone_args=None, pred_other=None):
        super().__init__()
        self.cfg = cfg
        self.odim = 1049
        self.ignore_id = -1
        self.backbone = E2E(self.odim, backbone_args)
        self.sos = self.odim - 1
        self.eos = self.odim - 1

    def get_encoded_features(self, video, audio, padding_mask):
        return self.backbone.encoder(video, audio, padding_mask)


class USR(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.model = USRModel(cfg, cfg.model.backbone)
        self.cfg = cfg
