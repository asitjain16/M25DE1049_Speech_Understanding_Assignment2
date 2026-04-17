"""
Part 3: Zero-Shot Cross-Lingual Voice Cloning (TTS)
Implements speaker embedding extraction, prosody warping, and synthesis
"""

import torch
import numpy as np
from typing import Dict, Tuple
import logging
import librosa

from .speaker_encoder import SpeakerEncoder
from .prosody_transfer import ProsodyTransfer
from .tts_model import TTSModel

logger = logging.getLogger(__name__)


class SynthesisPipeline:
    """Complete TTS pipeline with zero-shot voice cloning"""
    
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        # Task 3.1: Initialize speaker encoder
        self.speaker_encoder = SpeakerEncoder(device=device)
        
        # Task 3.2: Initialize prosody transfer module
        self.prosody_transfer = ProsodyTransfer(
            use_dtw=config['prosody']['use_dtw'],
            f0_weight=config['prosody']['f0_weight'],
            energy_weight=config['prosody']['energy_weight'],
            dtw_window=config['prosody']['dtw_window']
        )
        
        # Task 3.3: Initialize TTS model
        self.tts_model = TTSModel(
            model_name=config['model'],
            device=device
        )
    
    def extract_speaker_embedding(self, reference_audio_path: str) -> torch.Tensor:
        """
        Task 3.1: Extract speaker embedding from 60s reference
        
        Args:
            reference_audio_path: Path to reference audio (60s)
            
        Returns:
            Speaker embedding tensor (d-vector or x-vector)
        """
        logger.info("Extracting speaker embedding...")
        
        # Load reference audio
        audio, sr = librosa.load(reference_audio_path, sr=16000, duration=60)
        
        # Verify duration
        duration = len(audio) / sr
        if duration < 59 or duration > 61:
            logger.warning(f"Reference audio duration {duration:.1f}s, expected 60s")
        
        # Extract embedding
        embedding = self.speaker_encoder.encode(audio, sr)
        
        logger.info(f"Speaker embedding shape: {embedding.shape}")
        return embedding
    
    def extract_and_warp_prosody(self, source_audio_path: str, 
                                 target_audio_path: str) -> Dict:
        """
        Task 3.2: Extract F0 and energy, apply DTW warping
        
        Args:
            source_audio_path: Original professor's lecture
            target_audio_path: Your reference voice
            
        Returns:
            Dictionary with warped prosody features
        """
        logger.info("Extracting and warping prosody features...")
        
        # Load audio files
        source_audio, sr = librosa.load(source_audio_path, sr=22050)
        target_audio, _ = librosa.load(target_audio_path, sr=22050)
        
        # Extract prosody from source
        source_prosody = self.prosody_transfer.extract_prosody(source_audio, sr)
        
        # Extract prosody from target (for reference)
        target_prosody = self.prosody_transfer.extract_prosody(target_audio, sr)
        
        # Apply DTW warping
        warped_prosody = self.prosody_transfer.warp_prosody(
            source_prosody,
            target_prosody
        )
        
        logger.info("Prosody warping complete")
        return warped_prosody
    
    def synthesize(self, text: str, speaker_embedding: torch.Tensor, 
                   prosody_features: Dict) -> np.ndarray:
        """
        Task 3.3: Synthesize speech in LRL with cloned voice
        
        Args:
            text: Text in target LRL
            speaker_embedding: Speaker embedding from reference
            prosody_features: Warped prosody features
            
        Returns:
            Synthesized audio array (22.05kHz or higher)
        """
        logger.info("Synthesizing speech...")
        
        # Synthesize with TTS model
        audio = self.tts_model.synthesize(
            text=text,
            speaker_embedding=speaker_embedding,
            prosody=prosody_features
        )
        
        # Verify sample rate
        target_sr = self.config['output']['sample_rate']
        if target_sr < 22050:
            logger.warning(f"Output sample rate {target_sr} below 22.05kHz requirement")
        
        logger.info(f"Synthesis complete. Audio length: {len(audio)/target_sr:.2f}s")
        return audio
    
    def compute_mcd(self, synthesized_audio: np.ndarray, 
                    reference_audio_path: str) -> float:
        """
        Compute Mel-Cepstral Distortion between synthesized and reference
        
        Args:
            synthesized_audio: Synthesized audio array
            reference_audio_path: Path to reference audio
            
        Returns:
            MCD value (target: < 8.0)
        """
        logger.info("Computing Mel-Cepstral Distortion...")
        
        # Load reference
        reference_audio, sr = librosa.load(reference_audio_path, sr=22050)
        
        # Extract MFCCs
        synth_mfcc = librosa.feature.mfcc(y=synthesized_audio, sr=sr, n_mfcc=13)
        ref_mfcc = librosa.feature.mfcc(y=reference_audio, sr=sr, n_mfcc=13)
        
        # Align sequences (use DTW or truncate)
        min_len = min(synth_mfcc.shape[1], ref_mfcc.shape[1])
        synth_mfcc = synth_mfcc[:, :min_len]
        ref_mfcc = ref_mfcc[:, :min_len]
        
        # Compute MCD
        # MCD = (10/ln(10)) * sqrt(2 * sum((c_synth - c_ref)^2))
        diff = synth_mfcc - ref_mfcc
        mcd = (10.0 / np.log(10)) * np.sqrt(2 * np.mean(diff ** 2))
        
        logger.info(f"MCD: {mcd:.2f} (target: < 8.0)")
        
        if mcd >= 8.0:
            logger.warning(f"MCD {mcd:.2f} exceeds threshold 8.0")
        
        return mcd
