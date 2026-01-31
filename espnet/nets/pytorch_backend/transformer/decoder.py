#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Decoder definition."""

from typing import Any
from typing import List
from typing import Tuple

import torch

from espnet.nets.pytorch_backend.nets_utils import rename_state_dict
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.decoder_layer import DecoderLayer
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.scorer_interface import BatchScorerInterface


def _pre_hook(
    state_dict,
    prefix,
    local_metadata,
    strict,
    missing_keys,
    unexpected_keys,
    error_msgs,
):
    # https://github.com/espnet/espnet/commit/3d422f6de8d4f03673b89e1caef698745ec749ea#diff-bffb1396f038b317b2b64dd96e6d3563
    rename_state_dict(prefix + "output_norm.", prefix + "after_norm.", state_dict)


class Decoder(BatchScorerInterface, torch.nn.Module):
    """Transfomer decoder module.
    :param int idim: input/vocab dim
    :param int attention_dim: dimention of attention
    :param int attention_heads: the number of heads of multi head attention
    :param int linear_units: the number of units of position-wise feed forward
    :param int num_blocks: the number of decoder blocks
    :param float dropout_rate: dropout rate
    :param float positional_dropout_rate: dropout rate for positional encoding
    :param float self_attention_dropout_rate: dropout rate for self attention
    :param float src_attention_dropout_rate: dropout rate for source attention
    :param torch.nn.Module proj_decoder: optional projection applied to encoder memory
    """

    def __init__(
        self,
        idim,
        attention_dim=256,
        attention_heads=4,
        linear_units=2048,
        num_blocks=6,
        dropout_rate=0.1,
        positional_dropout_rate=0.1,
        self_attention_dropout_rate=0.0,
        src_attention_dropout_rate=0.0,
        proj_decoder=None,
    ):
        """Construct a Decoder object."""
        torch.nn.Module.__init__(self)
        self._register_load_state_dict_pre_hook(_pre_hook)

        self.embed = torch.nn.Sequential(
            torch.nn.Embedding(idim, attention_dim),
            PositionalEncoding(attention_dim, positional_dropout_rate),
        )
        self.decoders = repeat(
            num_blocks,
            lambda: DecoderLayer(
                attention_dim,
                MultiHeadedAttention(
                    attention_heads, attention_dim, self_attention_dropout_rate
                ),
                MultiHeadedAttention(
                    attention_heads, attention_dim, src_attention_dropout_rate
                ),
                PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate,
            ),
        )
        self.after_norm = LayerNorm(attention_dim)

        self.adim = attention_dim

        self.out_layer_v = torch.nn.Linear(attention_dim, idim)
        self.out_layer_a = torch.nn.Linear(attention_dim, idim)
        self.out_layer_av = torch.nn.Linear(attention_dim, idim)

        self.proj_decoder = proj_decoder

        self.apply(self._init_weights)

    def forward(self, tgt, tgt_mask, memory, memory_mask):
        """Forward decoder.
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param torch.Tensor memory_mask: encoded memory mask,  (batch, maxlen_in)
        :return x: decoded token score before softmax (batch, maxlen_out, token)
        :rtype: torch.Tensor
        :return tgt_mask: score mask before softmax (batch, maxlen_out)
        :rtype: torch.Tensor
        """
        if self.proj_decoder:
            memory = self.proj_decoder(memory)

        x = self.embed(tgt)
        x, tgt_mask, memory, memory_mask = self.decoders(
            x, tgt_mask, memory, memory_mask
        )
        x = self.after_norm(x)
        return x, tgt_mask

    def forward_one_step(self, tgt, tgt_mask, memory, memory_mask=None, cache=None):
        """Forward one step.
        :param torch.Tensor tgt: input token ids, int64 (batch, maxlen_out)
        :param torch.Tensor tgt_mask: input token mask,  (batch, maxlen_out)
        :param torch.Tensor memory: encoded memory, float32  (batch, maxlen_in, feat)
        :param List[torch.Tensor] cache:
            cached output list of (batch, max_time_out-1, size)
        :return y, cache: NN output value and cache per `self.decoders`.
        :rtype: Tuple[torch.Tensor, List[torch.Tensor]]
        """
        if self.proj_decoder:
            memory = self.proj_decoder(memory)

        x = self.embed(tgt)
        if cache is None:
            cache = [None] * len(self.decoders)
        new_cache = []
        for c, decoder in zip(cache, self.decoders):
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c,
            )
            new_cache.append(x)

        y = self.after_norm(x[:, -1])
        return y, new_cache

    # beam search API (see ScorerInterface)
    def score(self, ys, state, x, modality):
        """Score."""
        ys_mask = subsequent_mask(len(ys), device=x.device).unsqueeze(0)
        ys, state = self.forward_one_step(
            ys.unsqueeze(0), ys_mask, x.unsqueeze(0), cache=state
        )
        if modality == "v":
            out_layer = self.out_layer_v
        elif modality == "a":
            out_layer = self.out_layer_a
        else:
            out_layer = self.out_layer_av
        logp = torch.log_softmax(out_layer(ys), dim=-1)
        return logp.squeeze(0), state

    # batch beam search API (see BatchScorerInterface)
    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor, modality: str
    ) -> Tuple[torch.Tensor, List[Any]]:
        """Score new token batch (required).
        Args:
            ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
            states (List[Any]): Scorer states for prefix tokens.
            xs (torch.Tensor):
                The encoder feature that generates ys (n_batch, xlen, n_feat).
        Returns:
            tuple[torch.Tensor, List[Any]]: Tuple of
                batchfied scores for next token with shape of `(n_batch, n_vocab)`
                and next state list for ys.
        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoders)
        if states[0] is None:
            batch_state = None
        else:
            # transpose state of [batch, layer] into [layer, batch]
            batch_state = [
                torch.stack([states[b][l] for b in range(n_batch)])
                for l in range(n_layers)
            ]

        # batch decoding
        ys_mask = subsequent_mask(ys.size(-1), device=xs.device).unsqueeze(0)
        ys, states = self.forward_one_step(ys, ys_mask, xs, cache=batch_state)

        if modality == "v":
            out_layer = self.out_layer_v
        elif modality == "a":
            out_layer = self.out_layer_a
        else:
            out_layer = self.out_layer_av

        logp = torch.log_softmax(out_layer(ys), dim=-1)

        # transpose state of [layer, batch] into [batch, layer]
        state_list = [[states[l][b] for l in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

    def _init_weights(self, m):
        if isinstance(m, torch.nn.Linear):
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
