"""
USR 2.0 — Extract encoder features from a video.

Saves a dictionary with keys ``video``, ``audio``, and ``audio_visual``,
each mapping to a numpy array of shape (T', D) where T' is the number of
encoder output frames and D is the model dimension.

Usage
-----
    python extract_features.py \
        video=path/to/video.mp4 \
        model.pretrained_model_path=path/to/model.pth \
        output=features.pt

Optionally save only a subset of modalities:
    python extract_features.py \
        video=path/to/video.mp4 \
        model.pretrained_model_path=path/to/model.pth \
        modality=av \
        output=features.pt

Override backbone:
    python extract_features.py ... model/backbone=resnet_transformer_large
"""

import logging
import sys

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from demo import load_model, load_video_audio, preprocess_video
from preprocessing.landmarks_detector import LandmarksDetector
from preprocessing.video_preprocess import VideoProcess
from utils.utils import UNIGRAM1000_LIST

log = logging.getLogger(__name__)

OmegaConf.register_new_resolver("len", len, replace=True)


@torch.no_grad()
def extract(video_path: str, cfg: DictConfig, modality: str = "av",
            device: torch.device = torch.device("cuda")):
    """Extract encoder features and return a dict of numpy arrays."""

    video_frames, audio = load_video_audio(video_path)

    log.info("Detecting landmarks and cropping mouth region ...")
    ld = LandmarksDetector(device=str(device), model_name="mobilenet0.25")
    vp = VideoProcess(convert_gray=False)
    video_tensor = preprocess_video(video_frames, ld, vp)

    log.info("Loading model from: %s", cfg.model.pretrained_model_path)
    model = load_model(cfg, cfg.model.pretrained_model_path, device)

    video_input = video_tensor.unsqueeze(0).to(device)
    audio_input = audio.unsqueeze(0).to(device).transpose(1, 2)

    features = {}
    modality = modality.lower()

    if modality in ("av", "all"):
        feat_v = model.encoder(xs_v=video_input)
        feat_a = model.encoder(xs_a=audio_input)
        feat_av = model.encoder(xs_v=video_input, xs_a=audio_input)
        features["video"] = feat_v.squeeze(0).cpu().numpy()
        features["audio"] = feat_a.squeeze(0).cpu().numpy()
        features["audio_visual"] = feat_av.squeeze(0).cpu().numpy()
    elif modality == "v" or modality == "video":
        feat = model.encoder(xs_v=video_input)
        features["video"] = feat.squeeze(0).cpu().numpy()
    elif modality == "a" or modality == "audio":
        feat = model.encoder(xs_a=audio_input)
        features["audio"] = feat.squeeze(0).cpu().numpy()
    else:
        raise ValueError(f"Unknown modality '{modality}'.")

    return features


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    video_path = cfg.get("video")
    checkpoint = cfg.get("model", {}).get("pretrained_model_path")
    output_path = cfg.get("output", "features.pt")
    modality = cfg.get("modality", "av")

    if not video_path:
        print("Error: video=path/to/video.mp4 is required.")
        sys.exit(1)
    if not checkpoint:
        print("Error: model.pretrained_model_path=path/to/model.pth is required.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features = extract(video_path, cfg, modality=modality, device=device)

    torch.save(features, output_path)
    print(f"\nSaved features to: {output_path}")
    for k, v in features.items():
        print(f"  {k:15s} : shape {v.shape}, dtype {v.dtype}")


if __name__ == "__main__":
    main()
