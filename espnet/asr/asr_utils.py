# Copyright 2017 Johns Hopkins University (Shinji Watanabe)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging

import torch


def torch_load(path, model):
    """Load torch model states.

    Args:
        path (str): Model path or snapshot file path to be loaded.
        model (torch.nn.Module): Torch model.

    """
    if "snapshot" in path:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)[
            "model"
        ]
    else:
        model_state_dict = torch.load(path, map_location=lambda storage, loc: storage)

    if hasattr(model, "module"):
        model.module.load_state_dict(model_state_dict)
    else:
        model.load_state_dict(model_state_dict)

    del model_state_dict


# * ------------------ recognition related ------------------ *
def parse_hypothesis(hyp, char_list):
    """Parse hypothesis.

    Args:
        hyp (list[dict[str, Any]]): Recognition hypothesis.
        char_list (list[str]): List of characters.

    Returns:
        tuple(str, str, str, float)

    """
    # remove sos and get results
    tokenid_as_list = list(map(int, hyp["yseq"][1:]))
    token_as_list = [char_list[idx] for idx in tokenid_as_list]
    score = float(hyp["score"])

    # convert to string
    tokenid = " ".join([str(idx) for idx in tokenid_as_list])
    token = " ".join(token_as_list)
    text = "".join(token_as_list).replace("<space>", " ")

    return text, token, tokenid, score


def add_results_to_json(nbest_hyps, char_list):
    """Add N-best results to json.
    Args:
        nbest_hyps (list[dict[str, Any]]): List of hypotheses.
        char_list (list[str]): List of characters.
    Returns:
        str: 1-best result
    """
    assert len(nbest_hyps) == 1, "only 1-best result is supported."
    # parse hypothesis
    rec_text, rec_token, rec_tokenid, score = parse_hypothesis(nbest_hyps[0], char_list)
    return rec_text
