# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Transformer speech recognition model (pytorch)."""

import torch

from espnet.nets.pytorch_backend.ctc import CTC
from espnet.nets.pytorch_backend.transformer.decoder import Decoder
from espnet.nets.pytorch_backend.transformer.encoder_av import Encoder
from espnet.nets.scorers.ctc import CTCPrefixScorer


class E2E(torch.nn.Module):
    def __init__(self, odim, args, ignore_id=-1):
        """Construct an E2E object.
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        torch.nn.Module.__init__(self)

        self.encoder = Encoder(
            idim=args.idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            gamma_init=getattr(args, "gamma_init", 0.1),
        )

        ctc_rel_weight = getattr(args, "ctc_rel_weight", 0.1)
        dropout_rate = 0.1

        if ctc_rel_weight > 0.0:
            self.ctc_v = CTC(odim, args.adim, dropout_rate, ctc_type="warpctc", reduce="sum")
            self.ctc_a = CTC(odim, args.adim, dropout_rate, ctc_type="warpctc", reduce="sum")
            self.ctc_av = CTC(odim, args.adim, dropout_rate, ctc_type="warpctc", reduce="sum")
        else:
            self.ctc_v = self.ctc_a = self.ctc_av = None

        if ctc_rel_weight < 1.0:
            self.decoder = Decoder(
                idim=1049,
                attention_dim=args.ddim,
                attention_heads=args.dheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                proj_decoder=torch.nn.Linear(args.adim, args.ddim) if args.adim != args.ddim else None,
            )
        else:
            self.decoder = None

        self.sos = odim - 1
        self.eos = odim - 1
        self.odim = odim
        self.ignore_id = ignore_id

        self.args = args

    def scorers(self):
        """Scorers."""
        ctc_scorer = CTCPrefixScorer(self.ctc_v, self.ctc_a, self.ctc_av, self.eos) if self.ctc_v is not None else None
        return dict(decoder=self.decoder, ctc=ctc_scorer)
