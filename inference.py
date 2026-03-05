#!/usr/bin/env python3
"""
Basque TTS inference script.

CLI usage
---------
    python inference.py \
        --config  Configs/config_basque_multispeaker_phoneme_wavlm_800_2nd_normal.yml \
        --model   Models/Basque_Multispeaker_Phoneme_wavlm_normal/epoch_2nd_00030.pth \
        --text    "Kaixo, zelan zaude?" \
        --ref     Demo/ref_antton.wav \
        --output  output/kaixo.wav

Programmatic usage
------------------
    from inference import Synthesizer

    synth = Synthesizer(
        config='Configs/config_basque_multispeaker_phoneme_wavlm_800_2nd_normal.yml',
        checkpoint='Models/Basque_Multispeaker_Phoneme_wavlm_normal/epoch_2nd_00030.pth',
        ref='Demo/ref_antton.wav',
    )
    wav = synth.run("Kaixo, zelan zaude?")
    synth.save(wav, "output/kaixo.wav")
"""

import os
import sys
import time
import argparse

import torch
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as TT
import soundfile as sf
import yaml
from collections import OrderedDict

# ── repo root on sys.path so we can import project modules regardless of cwd ──
_REPO_ROOT = os.path.abspath(os.path.dirname(__file__))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from utils import recursive_munch
from models import load_ASR_models, load_F0_models, build_model
from text_utils import TextCleaner
from Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule

try:
    from phonemizer.eu_phonemizer import Phonemizer as _EuPhonemizer
except Exception:
    _EuPhonemizer = None


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers (mirror the notebook's preprocessing utilities)
# ─────────────────────────────────────────────────────────────────────────────

def _load_and_trim(path: str, target_sr: int = 24000, top_db: int = 30):
    """Load a WAV file, convert to mono at `target_sr`, and trim silence."""
    wav, orig_sr = torchaudio.load(path)
    wav = wav.mean(0)                                   # stereo → mono
    if orig_sr != target_sr:
        wav = TT.Resample(orig_sr, target_sr)(wav)
    frame_length, hop_length = 2048, 512
    if len(wav) >= frame_length:
        frames    = wav.unfold(0, frame_length, hop_length)
        rms       = frames.pow(2).mean(-1).sqrt()
        ref       = rms.max()
        if ref > 0:
            threshold = ref * (10 ** (-top_db / 20.0))
            nonsilent = (rms > threshold).nonzero(as_tuple=False).squeeze(-1)
            if nonsilent.numel() > 0:
                start = nonsilent[0].item() * hop_length
                end   = min((nonsilent[-1].item() + 1) * hop_length + frame_length, len(wav))
                wav   = wav[start:end]
    return wav.numpy()


def _to_mel(wave, n_mels=80, n_fft=2048, win_length=1200, hop_length=300,
            mean=-4.0, std=4.0):
    transform   = TT.MelSpectrogram(n_mels=n_mels, n_fft=n_fft,
                                     win_length=win_length, hop_length=hop_length)
    wave_tensor = torch.from_numpy(wave).float()
    mel         = transform(wave_tensor)
    mel         = (torch.log(1e-5 + mel.unsqueeze(0)) - mean) / std
    return mel


def _length_to_mask(lengths: torch.Tensor):
    mask = (torch.arange(lengths.max())
            .unsqueeze(0)
            .expand(lengths.shape[0], -1)
            .type_as(lengths))
    return torch.gt(mask + 1, lengths.unsqueeze(1))


# ─────────────────────────────────────────────────────────────────────────────
# Synthesizer class
# ─────────────────────────────────────────────────────────────────────────────

class Synthesizer:
    """Load a StyleTTS2 Basque model once and synthesize on demand.

    Parameters
    ----------
    config : str
        Path to the YAML config file.
    checkpoint : str
        Path to the model checkpoint (.pth).
    ref : str, optional
        Default reference WAV for style conditioning.  Can be overridden
        per-call in `run()`.
    device : str
        'cuda' or 'cpu'.  Defaults to CUDA if available.
    """

    def __init__(self, config: str, checkpoint: str,
                 ref: str = None, device: str = None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self._ref_default = ref

        # ── load config ───────────────────────────────────────────────────────
        cfg = yaml.safe_load(open(config))
        self._model_params = recursive_munch(cfg['model_params'])
        _pp = cfg.get('preprocess_params', {})
        _sp = _pp.get('spect_params', {})
        self._mel_params = dict(
            n_mels     = (self._model_params.n_mels
                          if hasattr(self._model_params, 'n_mels') else 80),
            n_fft      = _sp.get('n_fft',      2048),
            win_length = _sp.get('win_length', 1200),
            hop_length = _sp.get('hop_length',  300),
            mean       = _pp.get('mean', -4.0),
            std        = _pp.get('std',   4.0),
        )
        self._sr = _pp.get('sr', 24000)

        # ── build & load model ────────────────────────────────────────────────
        ASR_path   = cfg.get('ASR_path',   False)
        ASR_config = cfg.get('ASR_config', False)
        ASR_module = cfg.get('ASR_module', None)
        text_aligner   = load_ASR_models(ASR_path, ASR_config, ASR_module)
        pitch_extractor = load_F0_models(cfg.get('F0_path', False))

        BERT_path = cfg.get('PLBERT_dir', False)
        if 'phoneme' in BERT_path:
            from Utils.PLBERT_phoneme.util import load_plbert
        elif 'subword' in BERT_path:
            from Utils.PLBERT_subword.util import load_plbert
        elif 'naive' in BERT_path:
            from Utils.PLBERT_naive.util import load_plbert
        else:
            from Utils.PLBERT_all_languages.util import load_plbert
        plbert = load_plbert(BERT_path)

        self._model = build_model(self._model_params, text_aligner,
                                  pitch_extractor, plbert)
        for key in self._model:
            self._model[key].eval().to(self.device)

        self._load_checkpoint(checkpoint)

        # ── diffusion sampler ─────────────────────────────────────────────────
        self._sampler = DiffusionSampler(
            self._model.diffusion.diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False,
        )

        # ── text cleaner + phonemizer ─────────────────────────────────────────
        self._textcleaner = TextCleaner()
        self._phon = None
        if _EuPhonemizer is not None:
            phon_bin  = os.path.join(_REPO_ROOT, 'phonemizer', 'modulo1y2', 'modulo1y2')
            phon_dict = os.path.join(_REPO_ROOT, 'phonemizer', 'dict')
            try:
                self._phon = _EuPhonemizer(
                    language='eu', symbol='ipa',
                    path_modulo1y2=phon_bin,
                    path_dicts=phon_dict,
                )
            except Exception as e:
                print(f'Warning: could not initialize Basque phonemizer: {e}')

        # ── pre-cache default style embedding ────────────────────────────────
        self._ref_cache: dict = {}
        if ref is not None:
            self._ref_cache[ref] = self._compute_style(ref)

    # ── private helpers ───────────────────────────────────────────────────────

    def _load_checkpoint(self, ckpt_path: str):
        params_whole = torch.load(ckpt_path, map_location='cpu')
        params = params_whole.get('net', params_whole)
        for key in self._model:
            if key not in params:
                continue
            try:
                self._model[key].load_state_dict(params[key])
            except Exception:
                new_state = OrderedDict(
                    (k[7:] if k.startswith('module.') else k, v)
                    for k, v in params[key].items()
                )
                self._model[key].load_state_dict(new_state, strict=False)
        for key in self._model:
            self._model[key].eval()

    def _compute_style(self, path: str) -> torch.Tensor:
        """Compute (and cache) a style embedding from a reference WAV."""
        if path in self._ref_cache:
            return self._ref_cache[path]
        audio      = _load_and_trim(path, target_sr=self._sr)
        mel_tensor = _to_mel(audio, **self._mel_params).to(self.device)
        if mel_tensor.size(-1) < 16:
            mel_tensor = F.pad(mel_tensor, (0, 16 - mel_tensor.size(-1)))
        with torch.no_grad():
            ref_s = self._model.style_encoder(mel_tensor.unsqueeze(1))
            ref_p = self._model.predictor_encoder(mel_tensor.unsqueeze(1))
        embedding = torch.cat([ref_s, ref_p], dim=1)
        self._ref_cache[path] = embedding
        return embedding

    def _phonemize(self, text: str) -> str:
        """Normalize and phonemize a Basque text string."""
        text = text.strip()
        if text and text[-1] not in '.!?;:':
            text += '.'

        if self._phon is None:
            return ' '.join(text.split())

        phonemes = None
        for cand in [text.replace('\u2026', '...'), text]:
            try:
                norm     = self._phon.normalize(cand)
                phonemes = self._phon.getPhonemes(norm, use_single_char=True)
            except UnicodeDecodeError:
                try:
                    safe     = cand.encode('ISO-8859-15', errors='replace').decode('ISO-8859-15')
                    norm     = self._phon.normalize(safe)
                    phonemes = self._phon.getPhonemes(norm, use_single_char=True)
                except Exception:
                    phonemes = None
            except Exception as e:
                print(f'Warning: phonemizer error: {e}')
                phonemes = None
            if phonemes:
                break

        if phonemes:
            phonemes = phonemes.replace('\n', ' ').replace(' | ', ' ').replace('|', ' ')
            phonemes = ' '.join(phonemes.split())
        else:
            phonemes = ' '.join(text.split())
        return phonemes

    # ── public API ────────────────────────────────────────────────────────────

    def run(self, text: str,
            ref: str = None,
            alpha: float = 0.3,
            beta: float = 0.7,
            diffusion_steps: int = 5,
            embedding_scale: float = 1.0) -> 'np.ndarray':
        """Synthesize `text` and return a numpy waveform at 24 000 Hz.

        Parameters
        ----------
        text : str
            Input Basque text.
        ref : str, optional
            Path to a reference WAV for style conditioning.
            Falls back to the `ref` given at construction time.
        alpha : float
            Timbre mixing (0 = fully reference, 1 = fully diffusion-sampled).
        beta : float
            Prosody mixing (same scale as alpha).
        diffusion_steps : int
            Number of ADPM2 diffusion steps.
        embedding_scale : float
            Classifier-free guidance scale (>1 → more expressive).

        Returns
        -------
        numpy.ndarray  —  waveform, shape (T,), sample rate 24 000 Hz.
        """
        ref_path = ref or self._ref_default
        if ref_path is None:
            raise ValueError('A reference WAV must be provided either at '
                             'construction time or in run().')

        ref_s    = self._compute_style(ref_path)
        phonemes = self._phonemize(text)
        tokens   = self._textcleaner(phonemes)
        tokens.insert(0, 0)
        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)

        with torch.no_grad():
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask     = _length_to_mask(input_lengths).to(self.device)

            t_en     = self._model.text_encoder(tokens, input_lengths, text_mask)
            bert_dur = self._model.bert(tokens, attention_mask=(~text_mask).int())
            d_en     = self._model.bert_encoder(bert_dur).transpose(-1, -2)

            s_pred = self._sampler(
                noise           = torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding       = bert_dur,
                embedding_scale = embedding_scale,
                features        = ref_s,
                num_steps       = diffusion_steps,
            ).squeeze(1)

            s   = s_pred[:, 128:]
            ref_vec = s_pred[:, :128]
            ref_vec = alpha * ref_vec + (1 - alpha) * ref_s[:, :128]
            s       = beta  * s       + (1 - beta)  * ref_s[:, 128:]

            d        = self._model.predictor.text_encoder(d_en, s, input_lengths, text_mask)
            x, _     = self._model.predictor.lstm(d)
            duration = self._model.predictor.duration_proj(x)
            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)
            if pred_dur.dim() == 0:
                pred_dur = pred_dur.unsqueeze(0)

            seq_len   = int(input_lengths.item())
            total_dur = int(pred_dur.sum().item())
            pred_aln  = torch.zeros(seq_len, total_dur)
            c = 0
            for i in range(seq_len):
                d_i = int(pred_dur[i].item())
                pred_aln[i, c:c + d_i] = 1
                c += d_i

            en  = d.transpose(-1, -2) @ pred_aln.unsqueeze(0).to(self.device)
            asr = t_en @ pred_aln.unsqueeze(0).to(self.device)
            if self._model_params.decoder.type == 'hifigan':
                for src, name in [(en, 'en'), (asr, 'asr')]:
                    shifted        = torch.zeros_like(src)
                    shifted[:, :, 0]  = src[:, :, 0]
                    shifted[:, :, 1:] = src[:, :, :-1]
                    if name == 'en':
                        en  = shifted
                    else:
                        asr = shifted

            F0_pred, N_pred = self._model.predictor.F0Ntrain(en, s)
            out = self._model.decoder(asr, F0_pred, N_pred,
                                      ref_vec.squeeze().unsqueeze(0))

        return out.squeeze().cpu().numpy()[..., :-50]

    @staticmethod
    def save(wav, path: str, sr: int = 24000):
        """Write a waveform to disk.

        Parameters
        ----------
        wav : numpy.ndarray
            Waveform returned by `run()`.
        path : str
            Output file path (e.g. 'output/hello.wav').
        sr : int
            Sample rate (default 24 000 Hz).
        """
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        sf.write(path, wav, sr)
        print(f'Saved → {path}')


# ─────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ─────────────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description='Basque TTS inference with StyleTTS2',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--config',  required=True,
                   help='Path to YAML config file')
    p.add_argument('--model',   required=True,
                   help='Path to model checkpoint (.pth)')
    p.add_argument('--text',    required=True,
                   help='Input text to synthesize')
    p.add_argument('--ref',     required=True,
                   help='Reference WAV file for style conditioning')
    p.add_argument('--output',  default='output/synth.wav',
                   help='Output WAV file path')
    p.add_argument('--device',  default=None,
                   help="'cuda' or 'cpu' (auto-detected if omitted)")
    p.add_argument('--alpha',   type=float, default=0.3,
                   help='Timbre mixing weight')
    p.add_argument('--beta',    type=float, default=0.7,
                   help='Prosody mixing weight')
    p.add_argument('--diffusion_steps', type=int, default=5,
                   help='Number of diffusion sampling steps')
    p.add_argument('--embedding_scale', type=float, default=1.0,
                   help='Classifier-free guidance scale')
    return p.parse_args()


if __name__ == '__main__':
    args = _parse_args()

    print(f'Loading model from {args.model} …')
    synth = Synthesizer(
        config     = args.config,
        checkpoint = args.model,
        ref        = args.ref,
        device     = args.device,
    )

    print(f'Synthesizing: "{args.text}"')
    t0  = time.time()
    wav = synth.run(
        args.text,
        alpha           = args.alpha,
        beta            = args.beta,
        diffusion_steps = args.diffusion_steps,
        embedding_scale = args.embedding_scale,
    )
    rtf = (time.time() - t0) / (len(wav) / 24000)
    print(f'RTF: {rtf:.3f}')

    synth.save(wav, args.output)
