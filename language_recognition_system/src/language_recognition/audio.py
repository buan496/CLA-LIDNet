from __future__ import annotations

import struct
import wave
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _decode_pcm(raw_bytes: bytes, sample_width: int) -> np.ndarray:
    if sample_width == 1:
        samples = np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32)
        return (samples - 128.0) / 128.0
    if sample_width == 2:
        return np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    if sample_width == 4:
        return np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
    raise ValueError(f"unsupported sample width: {sample_width}")


def _read_wav_manual(path: str | Path) -> Tuple[np.ndarray, int, int]:
    with open(path, "rb") as file:
        header = file.read(12)
        if len(header) != 12:
            raise ValueError("invalid wav header")

        riff, _, wave_tag = struct.unpack("<4sI4s", header)
        if riff != b"RIFF" or wave_tag != b"WAVE":
            raise ValueError("unsupported wav container")

        fmt_chunk: bytes | None = None
        data_chunk: bytes | None = None

        while True:
            chunk_header = file.read(8)
            if len(chunk_header) < 8:
                break

            chunk_id, chunk_size = struct.unpack("<4sI", chunk_header)
            if chunk_id == b"fmt ":
                chunk_data = file.read(chunk_size)
                fmt_chunk = chunk_data
            elif chunk_id == b"data":
                chunk_data = file.read(chunk_size)
                data_chunk = chunk_data
            else:
                file.seek(chunk_size, 1)

            if chunk_size % 2 == 1:
                file.seek(1, 1)

            if fmt_chunk is not None and data_chunk is not None:
                break

    if fmt_chunk is None or data_chunk is None:
        raise ValueError("missing fmt or data chunk in wav file")

    if len(fmt_chunk) < 16:
        raise ValueError("invalid fmt chunk")

    format_tag, channels, sample_rate, _, _, bits_per_sample = struct.unpack("<HHIIHH", fmt_chunk[:16])
    sample_width = bits_per_sample // 8

    if format_tag == 1:
        samples = _decode_pcm(data_chunk, sample_width)
    elif format_tag == 3:
        if sample_width != 4:
            raise ValueError(f"unsupported IEEE float sample width: {sample_width}")
        samples = np.frombuffer(data_chunk, dtype=np.float32).astype(np.float32)
    else:
        raise ValueError(f"unsupported wav format tag: {format_tag}")

    return samples, sample_rate, channels


def load_wav(path: str | Path, target_sr: int = 16000) -> Tuple[np.ndarray, int]:
    try:
        with wave.open(str(path), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            sample_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            raw = wav_file.readframes(frame_count)
        samples = _decode_pcm(raw, sample_width)
    except wave.Error:
        samples, sample_rate, channels = _read_wav_manual(path)

    if channels > 1 and samples.size:
        samples = samples.reshape(-1, channels).mean(axis=1)

    peak = float(np.max(np.abs(samples))) if samples.size else 0.0
    if peak > 0:
        samples = samples / peak

    if sample_rate != target_sr and samples.size:
        waveform = torch.from_numpy(samples).view(1, 1, -1)
        target_length = int(round(samples.shape[0] * target_sr / sample_rate))
        waveform = F.interpolate(waveform, size=target_length, mode="linear", align_corners=False)
        samples = waveform.view(-1).numpy()
        sample_rate = target_sr

    return samples.astype(np.float32), sample_rate


def pad_or_trim(samples: np.ndarray, target_num_samples: int) -> np.ndarray:
    if samples.shape[0] == target_num_samples:
        return samples
    if samples.shape[0] > target_num_samples:
        return samples[:target_num_samples]

    padded = np.zeros(target_num_samples, dtype=np.float32)
    padded[: samples.shape[0]] = samples
    return padded
