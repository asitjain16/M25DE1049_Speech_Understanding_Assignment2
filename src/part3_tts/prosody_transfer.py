"""
Prosody Transfer using Dynamic Time Warping (DTW)
Extracts F0 and energy contours and warps them to target
"""

import numpy as np
import librosa
from scipy.interpolate import interp1d
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class ProsodyTransfer:
    """
    Prosody transfer module using DTW
    Preserves teaching style from source to target
    """
    
    def __init__(self, use_dtw: bool = True, f0_weight: float = 0.7,
                 energy_weight: float = 0.3, dtw_window: int = 50):
        self.use_dtw = use_dtw
        self.f0_weight = f0_weight
        self.energy_weight = energy_weight
        self.dtw_window = dtw_window
    
    def extract_prosody(self, audio: np.ndarray, sr: int) -> Dict:
        """
        Extract prosodic features from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Dictionary with F0, energy, and duration features
        """
        # Extract fundamental frequency (F0)
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=sr
        )
        
        # Handle unvoiced frames (interpolate)
        f0 = self._interpolate_unvoiced(f0, voiced_flag)
        
        # Extract energy (RMS)
        energy = librosa.feature.rms(y=audio, frame_length=2048, hop_length=512)[0]
        
        # Extract duration features
        duration = len(audio) / sr
        
        # Compute statistics
        f0_mean = np.mean(f0[~np.isnan(f0)])
        f0_std = np.std(f0[~np.isnan(f0)])
        energy_mean = np.mean(energy)
        energy_std = np.std(energy)
        
        return {
            'f0': f0,
            'energy': energy,
            'duration': duration,
            'f0_mean': f0_mean,
            'f0_std': f0_std,
            'energy_mean': energy_mean,
            'energy_std': energy_std,
            'voiced_flag': voiced_flag
        }
    
    def warp_prosody(self, source_prosody: Dict, target_prosody: Dict) -> Dict:
        """
        Warp source prosody to match target speaker characteristics
        
        Args:
            source_prosody: Prosody from original lecture
            target_prosody: Prosody from reference voice
            
        Returns:
            Warped prosody features
        """
        logger.info("Warping prosody features...")
        
        if self.use_dtw:
            # Apply DTW alignment
            warped_f0 = self._dtw_warp(
                source_prosody['f0'],
                target_prosody['f0']
            )
            
            warped_energy = self._dtw_warp(
                source_prosody['energy'],
                target_prosody['energy']
            )
        else:
            # Simple linear interpolation
            warped_f0 = self._linear_warp(
                source_prosody['f0'],
                len(target_prosody['f0'])
            )
            
            warped_energy = self._linear_warp(
                source_prosody['energy'],
                len(target_prosody['energy'])
            )
        
        # Normalize to target speaker's range
        warped_f0 = self._normalize_f0(
            warped_f0,
            source_prosody['f0_mean'],
            source_prosody['f0_std'],
            target_prosody['f0_mean'],
            target_prosody['f0_std']
        )
        
        warped_energy = self._normalize_energy(
            warped_energy,
            source_prosody['energy_mean'],
            source_prosody['energy_std'],
            target_prosody['energy_mean'],
            target_prosody['energy_std']
        )
        
        return {
            'f0': warped_f0,
            'energy': warped_energy,
            'f0_weight': self.f0_weight,
            'energy_weight': self.energy_weight
        }
    
    def _dtw_warp(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """
        Apply Dynamic Time Warping to align sequences
        
        Args:
            source: Source sequence
            target: Target sequence
            
        Returns:
            Warped source sequence
        """
        try:
            from dtaidistance import dtw
            
            # Remove NaN values
            source_clean = source[~np.isnan(source)]
            target_clean = target[~np.isnan(target)]
            
            # Compute DTW path
            path = dtw.warping_path(source_clean, target_clean, window=self.dtw_window)
            
            # Warp source to target
            warped = np.zeros(len(target_clean))
            for i, j in path:
                if j < len(warped):
                    warped[j] = source_clean[i]
            
            return warped
            
        except ImportError:
            logger.warning("dtaidistance not available, using fastdtw")
            return self._fastdtw_warp(source, target)
    
    def _fastdtw_warp(self, source: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Fallback DTW using fastdtw"""
        try:
            from fastdtw import fastdtw
            
            source_clean = source[~np.isnan(source)].flatten()
            target_clean = target[~np.isnan(target)].flatten()
            
            # Reshape for fastdtw
            source_2d = source_clean.reshape(-1, 1)
            target_2d = target_clean.reshape(-1, 1)
            
            distance, path = fastdtw(source_2d, target_2d, radius=self.dtw_window)
            
            # Warp source to target
            warped = np.zeros(len(target_clean))
            for i, j in path:
                if j < len(warped):
                    warped[j] = source_clean[i]
            
            return warped
            
        except ImportError:
            logger.warning("fastdtw not available, using linear interpolation")
            return self._linear_warp(source, len(target))
    
    def _linear_warp(self, source: np.ndarray, target_length: int) -> np.ndarray:
        """Simple linear interpolation warping"""
        source_clean = source[~np.isnan(source)]
        
        # Create interpolation function
        x_old = np.linspace(0, 1, len(source_clean))
        x_new = np.linspace(0, 1, target_length)
        
        f = interp1d(x_old, source_clean, kind='cubic', fill_value='extrapolate')
        warped = f(x_new)
        
        return warped
    
    def _interpolate_unvoiced(self, f0: np.ndarray, voiced_flag: np.ndarray) -> np.ndarray:
        """Interpolate F0 values for unvoiced frames"""
        f0_interp = f0.copy()
        
        # Find voiced regions
        voiced_indices = np.where(voiced_flag)[0]
        
        if len(voiced_indices) > 1:
            # Interpolate unvoiced regions
            f = interp1d(
                voiced_indices,
                f0[voiced_indices],
                kind='linear',
                fill_value='extrapolate',
                bounds_error=False
            )
            
            all_indices = np.arange(len(f0))
            f0_interp = f(all_indices)
        
        return f0_interp
    
    def _normalize_f0(self, f0: np.ndarray, source_mean: float, source_std: float,
                      target_mean: float, target_std: float) -> np.ndarray:
        """Normalize F0 to target speaker's range"""
        # Z-score normalization and denormalization
        f0_normalized = (f0 - source_mean) / (source_std + 1e-8)
        f0_target = f0_normalized * target_std + target_mean
        
        return f0_target
    
    def _normalize_energy(self, energy: np.ndarray, source_mean: float, source_std: float,
                          target_mean: float, target_std: float) -> np.ndarray:
        """Normalize energy to target speaker's range"""
        energy_normalized = (energy - source_mean) / (source_std + 1e-8)
        energy_target = energy_normalized * target_std + target_mean
        
        return energy_target
