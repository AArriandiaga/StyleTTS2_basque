#!/usr/bin/env python3
"""
Simple test script adapted from `Colab/StyleTTS2_Demo_LibriTTS.ipynb`.

Features:
- Loads model and diffusion sampler from a checkpoint (auto-detect latest if not provided).
- Reads test examples from `Data/test_marina.txt` (phoneme inputs already provided). 
- Uses reference audio files from `/data/aholab/tts/eu/female/sonora/marina/` by default. 
- Writes generated WAVs to `output/test_outputs/`. 
Usage: python Demo/test_from_notebook.py --checkpoint PATH --test_list Data/test_marina.txt \ --ref_dir /data/aholab/tts/eu/female/sonora/marina/ --out_dir output/test_outputs/ """


import os
import argparse
import time
import yaml
from munch import Munch
import torch
import torchaudio
import librosa
import numpy as np
import soundfile as sf
import sys
# ensure repository root is on sys.path so imports from project root work
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from text_utils import TextCleaner
from models import load_ASR_models, load_F0_models, build_model
from utils import recursive_munch

def find_checkpoint(provided):
    # Auto-search removed: caller MUST provide an explicit checkpoint path.
    if provided and os.path.isfile(provided):
        return provided
    raise FileNotFoundError('You must provide --checkpoint pointing to a valid .pth file; auto-search is disabled.')


def length_to_mask(lengths: torch.Tensor):
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask+1, lengths.unsqueeze(1))
    return mask


def preprocess(wave, sr=24000, n_mels=80, n_fft=2048, win_length=1200, hop_length=300, mean=-4, std=4):
    to_mel = torchaudio.transforms.MelSpectrogram(n_mels=n_mels, n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    wave_tensor = torch.from_numpy(wave).float()
    mel_tensor = to_mel(wave_tensor)
    mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - mean) / std
    return mel_tensor


def compute_style(path, device, to_mel_params):
    wave, sr = librosa.load(path, sr=24000)
    audio, _ = librosa.effects.trim(wave, top_db=30)
    if sr != 24000:
        audio = librosa.resample(audio, sr, 24000)
    mel_tensor = preprocess(audio, **to_mel_params).to(device)
    with torch.no_grad():
        ref_s = model.style_encoder(mel_tensor.unsqueeze(1))
        ref_p = model.predictor_encoder(mel_tensor.unsqueeze(1))
    return torch.cat([ref_s, ref_p], dim=1)


def load_config(cfg_path):
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def safe_load_state(model, params):
    # load trained params with or without 'module.' prefix
    try:
        model.load_state_dict(params)
    except Exception:
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in params.items():
            name = k[7:] if k.startswith('module.') else k
            new_state[name] = v
        model.load_state_dict(new_state, strict=False)


def inference_from_tokens(tokens, ref_s, alpha=0.3, beta=0.7, diffusion_steps=5, embedding_scale=1.0):
    tokens = torch.LongTensor(tokens).to(device).unsqueeze(0)
    with torch.no_grad():
        input_lengths = torch.LongTensor([tokens.shape[-1]]).to(device)
        text_mask = length_to_mask(input_lengths).to(device)

        t_en = model.text_encoder(tokens, input_lengths, text_mask)
        bert_dur = model.bert(tokens, attention_mask=(~text_mask).int())
        d_en = model.bert_encoder(bert_dur).transpose(-1, -2)

        s_pred = sampler(noise = torch.randn((1, 256)).unsqueeze(1).to(device),
                         embedding=bert_dur,
                         embedding_scale=embedding_scale,
                         features=ref_s,
                         num_steps=diffusion_steps).squeeze(1)

        s = s_pred[:, 128:]
        ref = s_pred[:, :128]

        ref = alpha * ref + (1 - alpha)  * ref_s[:, :128]
        s = beta * s + (1 - beta)  * ref_s[:, 128:]

        d = model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
        x, _ = model.predictor.lstm(d)
        duration = model.predictor.duration_proj(x)
        duration = torch.sigmoid(duration).sum(axis=-1)
        pred_dur = torch.round(duration.squeeze()).clamp(min=1)

        pred_aln_trg = torch.zeros(input_lengths, int(pred_dur.sum().data))
        c_frame = 0
        for i in range(pred_aln_trg.size(0)):
            pred_aln_trg[i, c_frame:c_frame + int(pred_dur[i].data)] = 1
            c_frame += int(pred_dur[i].data)

        en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(en)
            asr_new[:, :, 0] = en[:, :, 0]
            asr_new[:, :, 1:] = en[:, :, 0:-1]
            en = asr_new

        F0_pred, N_pred = model.predictor.F0Ntrain(en, s)

        asr = (t_en @ pred_aln_trg.unsqueeze(0).to(device))
        if model_params.decoder.type == "hifigan":
            asr_new = torch.zeros_like(asr)
            asr_new[:, :, 0] = asr[:, :, 0]
            asr_new[:, :, 1:] = asr[:, :, 0:-1]
            asr = asr_new

        out = model.decoder(asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0))
    return out.squeeze().cpu().numpy()[..., :-50]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, default=None, help='path to checkpoint .pth')
    parser.add_argument('--config', type=str, default='Models/LibriTTS/config.yml', help='model config yml')
    parser.add_argument('--plbert_dir', type=str, default=None, help='override PLBERT_dir from config')
    parser.add_argument('--test_list', type=str, default='Data/test_marina.txt', help='test list (phoneme format)')
    parser.add_argument('--ref_file', type=str, required=True, help='reference audio file (.wav)')
    parser.add_argument('--out_dir', type=str, default='output/test_outputs/', help='where to save outputs')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    ckpt = find_checkpoint(args.checkpoint)
    if ckpt is None:
        raise FileNotFoundError('No checkpoint found; provide --checkpoint')
    print('Using checkpoint:', ckpt)

    cfg = load_config(args.config)
    # Convert nested model params to attribute-accessible structure (like training)
    model_params = recursive_munch(cfg['model_params'])

    device = args.device

    # load auxiliary models
    ASR_config = cfg.get('ASR_config', False)
    ASR_path = cfg.get('ASR_path', False)
    text_aligner = load_ASR_models(ASR_path, ASR_config) if ASR_path else None
    F0_path = cfg.get('F0_path', False)
    pitch_extractor = load_F0_models(F0_path) if F0_path else None

    # Load PLBERT following the same logic as the training script: select
    # the proper PLBERT util based on the `PLBERT_dir` string.
    # prefer CLI override if provided
    BERT_path = args.plbert_dir if args.plbert_dir is not None else cfg.get('PLBERT_dir', False)
    plbert = None
    if BERT_path:
        if 'phoneme' in BERT_path:
            from Utils.PLBERT_phoneme.util import load_plbert
        elif 'subword' in BERT_path:
            from Utils.PLBERT_subword.util import load_plbert
        elif 'naive' in BERT_path:
            from Utils.PLBERT_naive.util import load_plbert
        else:
            from Utils.PLBERT_all_languages.util import load_plbert
        plbert = load_plbert(BERT_path)

    # build model
    model = build_model(model_params, text_aligner, pitch_extractor, plbert)
    _ = [model[key].eval() for key in model]
    _ = [model[key].to(device) for key in model]

    # load checkpoint
    print('Loading checkpoint weights...')
    params_whole = torch.load(ckpt, map_location='cpu')
    params = params_whole.get('net', params_whole)
    for key in model:
        if key in params:
            print(f'Loading {key}')
            try:
                model[key].load_state_dict(params[key])
            except Exception:
                # try stripping module.
                safe_load_state(model[key], params[key])

    from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
    sampler = DiffusionSampler(
        model.diffusion.diffusion,
        sampler=ADPM2Sampler(),
        sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
        clamp=False,
    )

    # prepare text cleaner
    textcleaner = TextCleaner()

    # mel params for preprocess
    to_mel_params = dict(
        sr=24000,
        n_mels=model_params.n_mels if hasattr(model_params, 'n_mels') else 80,
        n_fft=cfg.get('preprocess_params', {}).get('spect_params', {}).get('n_fft', 2048),
        win_length=cfg.get('preprocess_params', {}).get('spect_params', {}).get('win_length', 1200),
        hop_length=cfg.get('preprocess_params', {}).get('spect_params', {}).get('hop_length', 300),
        mean=cfg.get('preprocess_params', {}).get('mean', -4) if cfg.get('preprocess_params') else -4,
        std=cfg.get('preprocess_params', {}).get('std', 4) if cfg.get('preprocess_params') else 4,
    )

    # ensure out dir
    os.makedirs(args.out_dir, exist_ok=True)

    # require a single reference file and validate it
    if not os.path.isfile(args.ref_file):
        raise FileNotFoundError(f'--ref_file {args.ref_file} not found or is not a file.')
    if not args.ref_file.lower().endswith('.wav'):
        raise ValueError('--ref_file must point to a .wav file')

    # import model globals for helper functions
    globals()['model'] = model
    globals()['model_params'] = model_params
    globals()['sampler'] = sampler
    globals()['device'] = device

    # read test list
    with open(args.test_list, 'r', encoding='utf-8') as f:
        lines = [l.strip() for l in f if l.strip()]

    # ensure provided reference file is not a test sample (avoid self-reference)
    test_basenames = set()
    for l in lines:
        parts = l.split('|')
        if len(parts) >= 1 and parts[0].strip():
            test_basenames.add(os.path.basename(parts[0].strip()))
    if os.path.basename(args.ref_file) in test_basenames:
        raise ValueError('Provided --ref_file basename matches one of the test entries; provide a different reference file.')

    for idx, line in enumerate(lines):
        # expected: relative_wav_path|phoneme_text|label
        parts = line.split('|')
        if len(parts) >= 2:
            relpath = parts[0]
            phonemes = parts[1]
        else:
            # fallback: whole line as phoneme string
            relpath = None
            phonemes = line

        # compute tokens using TextCleaner (characters -> indices)
        tokens = textcleaner(phonemes)
        tokens.insert(0, 0)

        # use the provided single reference file for all syntheses
        ref_path = args.ref_file

        print(f'[{idx+1}/{len(lines)}] Synthesizing; ref={os.path.basename(ref_path)}')
        ref_s = compute_style(ref_path, device, to_mel_params)

        start = time.time()
        wav = inference_from_tokens(tokens, ref_s, diffusion_steps=5, embedding_scale=1.0)
        rtf = (time.time() - start) / (len(wav) / 24000)
        print(f'RTF = {rtf:.4f}')

        # save
        out_name = os.path.splitext(os.path.basename(relpath if relpath else f'sample_{idx}'))[0]
        # make it explicit these files are generated (avoid name collision with original data)
        out_path = os.path.join(args.out_dir, out_name + '_synth.wav')
        sf.write(out_path, wav, 24000)
        print('Saved ->', out_path)

    print('Done. Outputs in', args.out_dir)
