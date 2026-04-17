"""
Audio utility functions
"""

import numpy as np
import soundfile as sf
import librosa
from typing import Tuple


def load_audio(path: str, sr: int = 22050, mono: bool = True) -> Tuple[np.ndarray, int]:
    """
    Load audio file
    
    Args:
        path: Path to audio file
        sr: Target sample rate
        mono: Convert to mono
        
    Returns:
        Tuple of (audio array, sample rate)
    """
    audio, orig_sr = librosa.load(path, sr=sr, mono=mono)
    return audio, sr


def save_audio(audio: np.ndarray, path: str, sr: int = 22050):
    """
    Save audio to file
    
    Args:
        audio: Audio array
        path: Output path
        sr: Sample rate
    """
    sf.write(path, audio, sr)


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """
    Normalize audio to target dB level
    
    Args:
        audio: Input audio
        target_db: Target dB level
        
    Returns:
        Normalized audio
    """
    # Compute current RMS
    rms = np.sqrt(np.mean(audio ** 2))
    
    # Compute target RMS
    target_rms = 10 ** (target_db / 20)
    
    # Scale audio
    if rms > 0:
        audio = audio * (target_rms / rms)
    
    return audio


def compute_snr(signal: np.ndarray, noise: np.ndarray) -> float:
    """
    Compute Signal-to-Noise Ratio
    
    Args:
        signal: Clean signal
        noise: Noise signal
        
    Returns:
        SNR in dB
    """
    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)
    
    snr = 10 * np.log10(signal_power / (noise_power + 1e-10))
    return snr


def trim_silence(audio: np.ndarray, sr: int, threshold_db: float = -40.0) -> np.ndarray:
    """
    Trim silence from beginning and end of audio
    
    Args:
        audio: Input audio
        sr: Sample rate
        threshold_db: Silence threshold in dB
        
    Returns:
        Trimmed audio
    """
    trimmed, _ = librosa.effects.trim(audio, top_db=-threshold_db)
    return trimmed
