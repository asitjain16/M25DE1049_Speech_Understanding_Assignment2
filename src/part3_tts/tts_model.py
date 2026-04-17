"""
TTS Model wrapper for zero-shot voice cloning
Supports VITS, YourTTS, or Meta MMS
Python 3.12 compatible with fallback options
"""

import torch
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)


class TTSModel:
    """
    TTS model wrapper for synthesis
    Supports multiple backends with Python 3.12 compatibility
    """
    
    def __init__(self, model_name: str, device: torch.device):
        self.model_name = model_name
        self.device = device
        self.backend = None
        
        # Initialize TTS model
        self._init_model()
    
    def _init_model(self):
        """Initialize TTS model with fallback options"""
        # Try Coqui TTS first
        try:
            from TTS.api import TTS
            
            logger.info(f"Loading TTS model: {self.model_name}")
            self.tts = TTS(self.model_name).to(self.device)
            self.backend = 'coqui'
            logger.info("TTS model loaded successfully (Coqui TTS)")
            return
            
        except ImportError:
            logger.warning("Coqui TTS not available, trying alternatives...")
        except Exception as e:
            logger.warning(f"Failed to load Coqui TTS: {e}, trying alternatives...")
        
        # Try pyttsx3 (offline)
        try:
            import pyttsx3
            self.engine = pyttsx3.init()
            self.backend = 'pyttsx3'
            logger.info("Using pyttsx3 for TTS (offline)")
            return
        except ImportError:
            logger.warning("pyttsx3 not available, trying gTTS...")
        except Exception as e:
            logger.warning(f"Failed to initialize pyttsx3: {e}")
        
        # Try gTTS (online)
        try:
            from gtts import gTTS
            self.backend = 'gtts'
            logger.info("Using Google TTS (requires internet)")
            return
        except ImportError:
            logger.error("No TTS backend available. Install: pip install pyttsx3 or pip install gTTS")
            raise ImportError("No TTS backend available")
    
    def synthesize(self, text: str, speaker_embedding: torch.Tensor,
                   prosody: Dict = None) -> np.ndarray:
        """
        Synthesize speech from text
        
        Args:
            text: Input text in target LRL
            speaker_embedding: Speaker embedding for voice cloning
            prosody: Optional prosody features
            
        Returns:
            Synthesized audio array
        """
        # Handle empty text
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided, using placeholder")
            text = "This is a synthesized speech sample."
        
        logger.info(f"Synthesizing text: {text[:50]}...")
        
        try:
            if self.backend == 'coqui':
                audio = self._synthesize_coqui(text, speaker_embedding, prosody)
            elif self.backend == 'pyttsx3':
                audio = self._synthesize_pyttsx3(text)
            elif self.backend == 'gtts':
                audio = self._synthesize_gtts(text)
            else:
                raise RuntimeError("No TTS backend initialized")
            
            return audio
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            raise
    
    def _synthesize_coqui(self, text: str, speaker_embedding: torch.Tensor,
                         prosody: Dict = None) -> np.ndarray:
        """Synthesize using Coqui TTS"""
        # Convert embedding to numpy if needed
        if isinstance(speaker_embedding, torch.Tensor):
            speaker_embedding = speaker_embedding.cpu().numpy()
        
        # Synthesize with speaker embedding
        audio = self.tts.tts(
            text=text,
            speaker_embedding=speaker_embedding
        )
        
        # Apply prosody if provided
        if prosody is not None:
            audio = self._apply_prosody(audio, prosody)
        
        # Convert to numpy array
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        return audio
    
    def _synthesize_pyttsx3(self, text: str) -> np.ndarray:
        """Synthesize using pyttsx3 (offline)"""
        import tempfile
        import os
        
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required for pyttsx3 backend")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            temp_path = f.name
        
        try:
            # Synthesize to file
            self.engine.save_to_file(text, temp_path)
            self.engine.runAndWait()
            
            # Load audio
            audio, sr = librosa.load(temp_path, sr=22050)
            
            return audio
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _synthesize_gtts(self, text: str) -> np.ndarray:
        """Synthesize using Google TTS (online)"""
        from gtts import gTTS
        import tempfile
        import os
        
        try:
            import librosa
        except ImportError:
            raise ImportError("librosa required for gTTS backend")
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as f:
            temp_path = f.name
        
        try:
            # Synthesize to file
            tts = gTTS(text=text, lang='en')
            tts.save(temp_path)
            
            # Load audio
            audio, sr = librosa.load(temp_path, sr=22050)
            
            return audio
        finally:
            # Clean up
            if os.path.exists(temp_path):
                os.unlink(temp_path)
    
    def _apply_prosody(self, audio: np.ndarray, prosody: Dict) -> np.ndarray:
        """
        Apply prosody features to synthesized audio
        
        Args:
            audio: Synthesized audio
            prosody: Prosody features (F0, energy)
            
        Returns:
            Audio with applied prosody
        """
        import librosa
        
        # Extract current F0
        current_f0, _, _ = librosa.pyin(
            audio,
            fmin=librosa.note_to_hz('C2'),
            fmax=librosa.note_to_hz('C7'),
            sr=22050
        )
        
        # Get target F0
        target_f0 = prosody['f0']
        
        # Resample target F0 to match audio length
        if len(target_f0) != len(current_f0):
            from scipy.interpolate import interp1d
            x_old = np.linspace(0, 1, len(target_f0))
            x_new = np.linspace(0, 1, len(current_f0))
            f = interp1d(x_old, target_f0, kind='linear', fill_value='extrapolate')
            target_f0 = f(x_new)
        
        # Apply F0 modification using PSOLA or similar
        # Simplified: use librosa's pitch shift
        f0_ratio = np.nanmean(target_f0) / (np.nanmean(current_f0) + 1e-8)
        n_steps = 12 * np.log2(f0_ratio)
        
        audio_modified = librosa.effects.pitch_shift(
            audio,
            sr=22050,
            n_steps=n_steps
        )
        
        return audio_modified
