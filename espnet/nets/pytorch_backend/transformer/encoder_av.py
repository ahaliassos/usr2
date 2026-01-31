#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""

import torch
import torch.nn as nn

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transformer.attention import RelPositionMultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import RelPositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import PositionwiseFeedForward
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.backbones.conv3d_extractor import Conv3dResNet
from espnet.nets.pytorch_backend.backbones.conv1d_extractor import Conv1dResNet


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/21d70286c354c66c0350e65dc098d2ee236faccc#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "input_layer.", prefix + "embed.", state_dict)
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "norm.", prefix + "after_norm.", state_dict)


class Encoder(torch.nn.Module):
    """Transformer encoder module.

    :param int idim: input dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of encoder blocks
    :param float dropout_rate: dropout rate
    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        gamma_init=0.1,
    ):
        """Construct an Encoder object."""
        super(Encoder, self).__init__()
        self._register_load_state_dict_pre_hook(_pre_hook)

        # -- frontend module.
        self.frontend_a = Conv1dResNet(relu_type="swish")
        self.frontend_v = Conv3dResNet(relu_type="swish")

        # -- backend module.
        self.embed = RelPositionalEncoding(attention_dim, dropout_rate)
        self.normalize_before = True

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (attention_dim, linear_units, dropout_rate)

        encoder_attn_layer = RelPositionMultiHeadedAttention
        encoder_attn_layer_args = (
            attention_heads,
            attention_dim,
            dropout_rate,
            False,  # zero_triu
        )

        self.encoders = repeat(
            num_blocks,
            lambda: EncoderLayer(
                attention_dim,
                encoder_attn_layer(*encoder_attn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                gamma_init,
            ),
        )
        self.after_norm = LayerNorm(attention_dim)

        self.linear_a = nn.Linear(idim, attention_dim)
        self.linear_v = nn.Linear(idim, attention_dim)
        self.linear_av = nn.Linear(2*idim, attention_dim)

    def forward(self, xs_v=None, xs_a=None, masks=None):
        """Encode input sequence.

        :param torch.Tensor xs_v: video input tensor (optional)
        :param torch.Tensor xs_a: audio input tensor (optional)
        :param torch.Tensor masks: input mask
        :return: encoded tensor
        :rtype: torch.Tensor
        """
        assert xs_v is not None or xs_a is not None

        if xs_v is not None:
            xs_v = self.frontend_v(xs_v)
        if xs_a is not None:
            xs_a = self.frontend_a(xs_a)

        if xs_v is not None and xs_a is not None:
            xs = self.linear_av(torch.cat([xs_v, xs_a], dim=-1))
        else:
            xs = self.linear_v(xs_v) if xs_v is not None else self.linear_a(xs_a)

        if masks is None:
            masks = torch.ones(xs.shape[0], 1, xs.shape[1], dtype=torch.bool, device=xs.device)

        xs = self.embed(xs)
        xs, masks = self.encoders(xs, masks)

        if isinstance(xs, tuple):
            xs = xs[0]

        xs = self.after_norm(xs)

        return xs
