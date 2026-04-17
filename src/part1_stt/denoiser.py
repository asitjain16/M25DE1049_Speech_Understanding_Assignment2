"""
Audio Denoising using DeepFilterNet or Spectral Subtraction
Handles classroom background noise and reverb
"""

import numpy as np
import torch
import librosa
import soundfile as sf
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class AudioDenoiser:
    """
    Audio denoising for classroom recordings
    Supports DeepFilterNet and Spectral Subtraction
    """
    
    def __init__(self, method: str = "deepfilternet", strength: float = 0.7):
        self.method = method
        self.strength = strength
        
        if method == "deepfilternet":
            self._init_deepfilternet()
        
    def _init_deepfilternet(self):
        """Initialize DeepFilterNet model"""
        try:
            from df.enhance import enhance, init_df
            self.df_model, self.df_state, _ = init_df()
            logger.info("DeepFilterNet initialized")
        except ImportError:
            logger.warning("DeepFilterNet not available, falling back to spectral subtraction")
            self.method = "spectral_subtraction"
    
    def process(self, audio_path: str, sr: int = 22050) -> np.ndarray:
        """
        Denoise audio file
        
        Args:
            audio_path: Path to input audio
            sr: Target sample rate
            
        Returns:
            Denoised audio array
        """
        # Load audio
        audio, orig_sr = librosa.load(audio_path, sr=sr, mono=True)
        
        logger.info(f"Denoising with {self.method}...")
        
        if self.method == "deepfilternet":
            denoised = self._deepfilternet_denoise(audio, sr)
        else:
            denoised = self._spectral_subtraction(audio, sr)
        
        return denoised
    
    def _deepfilternet_denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Denoise using DeepFilterNet"""
        try:
            from df.enhance import enhance
            
            # Convert to tensor
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            
            # Enhance
            enhanced = enhance(self.df_model, self.df_state, audio_tensor)
            
            # Convert back to numpy
            denoised = enhanced.squeeze().numpy()
            
            return denoised
            
        except Exception as e:
            logger.error(f"DeepFilterNet failed: {e}, using spectral subtraction")
            return self._spectral_subtraction(audio, sr)
    
    def _spectral_subtraction(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Spectral subtraction denoising
        
        Algorithm:
        1. Estimate noise spectrum from silent regions
        2. Subtract noise spectrum from signal spectrum
        3. Apply over-subtraction factor and spectral floor
        4. Reconstruct time-domain signal
        """
        # STFT parameters
        n_fft = 2048
        hop_length = 512
        
        # Compute STFT
        stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        phase = np.angle(stft)
        
        # Estimate noise spectrum from first 0.5 seconds (assumed silence)
        noise_frames = int(0.5 * sr / hop_length)
        noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
        
        # Spectral subtraction parameters
        alpha = self.strength  # Over-subtraction factor
        beta = 0.02  # Spectral floor
        
        # Subtract noise
        magnitude_denoised = magnitude - alpha * noise_spectrum
        
        # Apply spectral floor
        magnitude_denoised = np.maximum(magnitude_denoised, beta * magnitude)
        
        # Reconstruct signal
        stft_denoised = magnitude_denoised * np.exp(1j * phase)
        audio_denoised = librosa.istft(stft_denoised, hop_length=hop_length)
        
        return audio_denoised
    
    def reduce_reverb(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """
        Reduce reverberation using inverse filtering
        
        Args:
            audio: Input audio
            sr: Sample rate
            
        Returns:
            Dereverberated audio
        """
        # Estimate room impulse response
        # Simplified approach: high-pass filter to reduce late reflections
        
        # Design high-pass filter
        cutoff = 100  # Hz
        nyquist = sr / 2
        normalized_cutoff = cutoff / nyquist
        
        b, a = signal.butter(4, normalized_cutoff, btype='high')
        
        # Apply filter
        audio_filtered = signal.filtfilt(b, a, audio)
        
        return audio_filtered
    
    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range"""
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val * 0.95  # Leave some headroom
        return audio
