from __future__ import annotations

import torch


def hz_to_mel(freq: torch.Tensor) -> torch.Tensor:
    return 2595.0 * torch.log10(1.0 + freq / 700.0)


def mel_to_hz(mel: torch.Tensor) -> torch.Tensor:
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def create_mel_filterbank(
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    f_min: float = 0.0,
    f_max: float | None = None,
) -> torch.Tensor:
    f_max = f_max if f_max is not None else sample_rate / 2
    mel_min = hz_to_mel(torch.tensor(float(f_min)))
    mel_max = hz_to_mel(torch.tensor(float(f_max)))
    mel_points = torch.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = mel_to_hz(mel_points)
    fft_bins = torch.floor((n_fft + 1) * hz_points / sample_rate).long()

    filterbank = torch.zeros(n_mels, n_fft // 2 + 1, dtype=torch.float32)
    for mel_idx in range(1, n_mels + 1):
        left = fft_bins[mel_idx - 1].item()
        center = max(fft_bins[mel_idx].item(), left + 1)
        right = max(fft_bins[mel_idx + 1].item(), center + 1)

        for bin_idx in range(left, center):
            filterbank[mel_idx - 1, bin_idx] = (bin_idx - left) / max(center - left, 1)
        for bin_idx in range(center, right):
            filterbank[mel_idx - 1, bin_idx] = (right - bin_idx) / max(right - center, 1)

    return filterbank


class LogMelSpectrogram:
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        win_length: int = 400,
        n_mels: int = 80,
    ) -> None:
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self._window_cache: dict[str, torch.Tensor] = {}
        self._mel_filterbank_cache: dict[str, torch.Tensor] = {}

    def _get_window(self, device: torch.device) -> torch.Tensor:
        device_key = str(device)
        if device_key not in self._window_cache:
            self._window_cache[device_key] = torch.hann_window(self.win_length, device=device)
        return self._window_cache[device_key]

    def _get_mel_filterbank(self, device: torch.device) -> torch.Tensor:
        device_key = str(device)
        if device_key not in self._mel_filterbank_cache:
            self._mel_filterbank_cache[device_key] = create_mel_filterbank(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
            ).to(device)
        return self._mel_filterbank_cache[device_key]

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.dim() != 1:
            raise ValueError("waveform must be a 1-D tensor")

        window = self._get_window(waveform.device)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            return_complex=True,
        )
        power_spec = stft.abs().pow(2.0)
        mel_filterbank = self._get_mel_filterbank(waveform.device)
        mel_spec = torch.matmul(mel_filterbank, power_spec)
        log_mel = torch.log(torch.clamp(mel_spec, min=1e-6))
        mean = log_mel.mean()
        std = log_mel.std().clamp(min=1e-6)
        return (log_mel - mean) / std
