import os

import cv2
import hydra
import numpy as np
import pytorchvideo
import torch
import torchaudio
import torchvision
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Grayscale,
    Lambda,
)

from data.transforms import NormalizeVideo, AddNoise
from espnet.asr.asr_utils import add_results_to_json, torch_load
from espnet.nets.batch_beam_search import BatchBeamSearch
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E
from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.scorers.length_bonus import LengthBonus
from preprocessing.landmarks_detector import LandmarksDetector
from preprocessing.video_preprocess import VideoProcess
from utils.utils import UNIGRAM1000_LIST


def video_transform():
    transform = [
        Lambda(lambda x: x / 255.), 
        CenterCrop(88), 
        Lambda(lambda x: x.transpose(0, 1)), 
        Grayscale(), 
        Lambda(lambda x: x.transpose(0, 1)),
        NormalizeVideo(mean=(0.421,), std=(0.165,))
    ]

    return Compose(transform)

# def audio_transform(noise_path, decode):
#     transform = [
#         AddNoise(noise_path=noise_path, snr_target=getattr(decode, "snr_target", 9999))
#     ]

#     return Compose(transform)


def get_beam_search(cfg, model):
    token_list = UNIGRAM1000_LIST
    odim = len(token_list)

    scorers = model.scorers()

    scorers["length_bonus"] = LengthBonus(len(token_list))

    weights = dict(
        decoder=1.0 - cfg.decode.ctc_weight,
        ctc=cfg.decode.ctc_weight,
        length_bonus=cfg.decode.penalty,
    )
    beam_search = BatchBeamSearch(
        beam_size=cfg.decode.beam_size,
        vocab_size=len(token_list),
        weights=weights,
        scorers=scorers,
        sos=odim - 1,
        eos=odim - 1,
        token_list=token_list,
        pre_beam_score_key=None if cfg.decode.ctc_weight == 1.0 else "decoder",
    )

    return beam_search

def load_av(path, target_vfps=25, target_afps=16000):
    def cut_or_pad(data, size, dim=0):
        # Pad with zeros on the right if data is too short
        # assert abs(data.size(dim) - size) < 2000 
        if data.size(dim) < size:
            # assert False
            padding = size - data.size(dim)
            data = torch.from_numpy(np.pad(data, (0, padding), "constant"))
        # Cut from the right if data is too long
        elif data.size(dim) > size:
            data = data[:size]
        # Keep if data is exactly right
        assert data.size(dim) == size
        return data

    # Read video + audio
    video, audio, info = torchvision.io.read_video(path, pts_unit="sec")
    vfps = info["video_fps"]
    afps = info["audio_fps"]

    # -------------------------
    # Audio: stereo -> mono -> resample
    # audio: (num_samples, num_channels)
    audio = audio.mean(dim=0, keepdim=True)  # mono, shape (num_samples, 1)

    if afps != target_afps:
        resampler = torchaudio.transforms.Resample(orig_freq=afps, new_freq=target_afps)
        audio = resampler(audio)

    # -------------------------
    # Video: resample to target_vfps
    if abs(vfps - target_vfps) > 1e-3:  # if not already 25 fps
        num_frames = video.shape[0]
        duration_sec = num_frames / vfps
        new_num_frames = int(duration_sec * target_vfps)

        # Resample along time axis
        indices = torch.linspace(0, num_frames - 1, new_num_frames).long()
        video = video[indices]
    
    audio = audio.transpose(0, 1)
    audio = cut_or_pad(audio, video.size(0) * (target_afps // target_vfps))
    audio = audio.transpose(0, 1)

    return video, audio


@hydra.main(config_path="conf", config_name="config")
def main(cfg):
    landmarks_detector = LandmarksDetector(device="cuda:0", model_name="mobilenet0.25")
    video_preprocess = VideoProcess(convert_gray=False)

    # video, audio = load_av("/home/ahaliassos/MS1_F07_C3.mp4")
    # video, audio = load_av("/home/ahaliassos/MS1_F07_C3.mp4")
    video, audio = load_av("/fsx/users/halichz/repos/usr_github/AbVf_1rep.mp4")
    # video, audio = load_av("/home/ahaliassos/50007.mp4")
    # video, audio = load_av("/fsx/behavioural_computing_data/LRS3/LRS3_original_dataset/LRS3/test/GAMR5RDSLJo/00011.mp4")

    video = video.numpy()
    landmarks = landmarks_detector(video)
    video = video_preprocess(video, landmarks)

    # torchvision.io.write_video("/home/ahaliassos/MS1_F07_C3_processed.mp4", video, fps=fps)
    
    video = torch.from_numpy(video).permute((3, 0, 1, 2))  # TxHxWxC -> # CxTxHxW

    video = video_transform()(video)
    # audio = audio_transform(cfg.data.noise_path, cfg.decode)(audio)

    model = torch.compile(E2E(1049, cfg.model.backbone))

    ckpt = torch.load(cfg.model.pretrained_model_path, map_location=lambda storage, loc: storage)
    ckpt = {k.replace("model.backbone.", "", 1): v for k, v in ckpt.items() if ".backbone." in k}
    
    model.load_state_dict(ckpt)
    model.eval()

    # AV
    beam_search = get_beam_search(cfg, model)

    with torch.no_grad():
        feat, _, _ = model.encoder.forward_single(xs_v=video, xs_a=audio.unsqueeze(0).transpose(1, 2))
    
    nbest_hyps_av = beam_search(
            x=feat.squeeze(0),
            modality="av",
            maxlenratio=cfg.decode.maxlenratio,
            minlenratio=cfg.decode.minlenratio
        )

    for i in range(5):
        nbest_hyps = [
            h.asdict() for h in nbest_hyps_av[i: min(len(nbest_hyps_av), i+1)]
            # h.asdict() for h in nbest_hyps[3: min(len(nbest_hyps), 4)]
        ]

        transcription = add_results_to_json(nbest_hyps, UNIGRAM1000_LIST)
        transcription = transcription.replace("<eos>", "")
        transcription = transcription.replace("▁", " ").strip()

        print(f"AV transcription top {i+1}:", transcription)

    # A
    beam_search = get_beam_search(cfg, model)

    with torch.no_grad():
        feat, _, _ = model.encoder.forward_single(xs_a=audio.unsqueeze(0).transpose(1, 2))
    
    nbest_hyps_a = beam_search(
            x=feat.squeeze(0),
            modality="a",
            maxlenratio=cfg.decode.maxlenratio,
            minlenratio=cfg.decode.minlenratio
        )
    
    for i in range(5):
        nbest_hyps = [
            h.asdict() for h in nbest_hyps_a[i: min(len(nbest_hyps_a), i+1)]
        ]

        transcription = add_results_to_json(nbest_hyps, UNIGRAM1000_LIST)
        transcription = transcription.replace("<eos>", "")
        transcription = transcription.replace("▁", " ").strip()

        print(f"A transcription top {i+1}:", transcription)

    # V
    beam_search = get_beam_search(cfg, model)

    with torch.no_grad():
        feat, _, _ = model.encoder.forward_single(xs_v=video)
    
    nbest_hyps_v = beam_search(
            x=feat.squeeze(0),
            modality="v",
            maxlenratio=cfg.decode.maxlenratio,
            minlenratio=cfg.decode.minlenratio
        )
    
    for i in range(5):
        nbest_hyps = [
            h.asdict() for h in nbest_hyps_v[i: min(len(nbest_hyps_v), i+1)]
        ]

        transcription = add_results_to_json(nbest_hyps, UNIGRAM1000_LIST)
        transcription = transcription.replace("<eos>", "")
        transcription = transcription.replace("▁", " ").strip()

        print(f"V transcription top {i+1}:", transcription)


if __name__ == "__main__":
    main()


    # # write to mp4
    # torchvision.io.write_video(
    #     "/home/ahaliassos/test_mouth.mp4",            # filename
    #     video_tensor,            # (T, H, W, C), dtype=uint8
    #     fps=25                   # frames per second
    # )


    
