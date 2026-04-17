"""
Adversarial Attack Module
Implements FGSM and other attacks on LID system
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class AdversarialAttack:
    """
    Adversarial attack generator
    Tests robustness of LID system
    """
    
    def __init__(self, method: str = 'fgsm', epsilon_range: list = [0.001, 0.1],
                 target_snr: float = 40.0, device: torch.device = None):
        self.method = method
        self.epsilon_range = epsilon_range
        self.target_snr = target_snr
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def find_minimum_perturbation(self, audio: np.ndarray, sr: int, 
                                  lid_model) -> Dict:
        """
        Find minimum perturbation to flip LID prediction
        
        Args:
            audio: Input audio (5 seconds)
            sr: Sample rate
            lid_model: Language ID model to attack
            
        Returns:
            Dictionary with min_epsilon, SNR, and adversarial audio
        """
        logger.info("Finding minimum adversarial perturbation...")
        
        # Convert audio to tensor
        audio_tensor = torch.from_numpy(audio).float().to(self.device)
        
        # Get original prediction
        original_pred = self._get_lid_prediction(audio_tensor, lid_model)
        logger.info(f"Original prediction: {original_pred}")
        
        # Binary search for minimum epsilon
        min_eps = self.epsilon_range[0]
        max_eps = self.epsilon_range[1]
        
        best_epsilon = None
        best_adv_audio = None
        
        for _ in range(20):  # Binary search iterations
            epsilon = (min_eps + max_eps) / 2
            
            # Generate adversarial example
            adv_audio = self._generate_adversarial(
                audio_tensor, lid_model, epsilon, original_pred
            )
            
            # Check if prediction flipped
            adv_pred = self._get_lid_prediction(adv_audio, lid_model)
            
            if adv_pred != original_pred:
                # Success - try smaller epsilon
                max_eps = epsilon
                best_epsilon = epsilon
                best_adv_audio = adv_audio
            else:
                # Failed - try larger epsilon
                min_eps = epsilon
        
        if best_epsilon is None:
            logger.warning("Could not find adversarial perturbation in range")
            best_epsilon = max_eps
            best_adv_audio = audio_tensor
        
        # Compute SNR
        snr = self._compute_snr(audio_tensor, best_adv_audio)
        
        return {
            'min_epsilon': best_epsilon,
            'snr': snr,
            'original_pred': original_pred,
            'adversarial_pred': self._get_lid_prediction(best_adv_audio, lid_model),
            'adversarial_audio': best_adv_audio.cpu().numpy()
        }
    
    def _generate_adversarial(self, audio: torch.Tensor, model, 
                             epsilon: float, target_class: int) -> torch.Tensor:
        """
        Generate adversarial example using FGSM
        
        Args:
            audio: Input audio tensor
            model: Target model
            epsilon: Perturbation magnitude
            target_class: Original class to flip from
            
        Returns:
            Adversarial audio tensor
        """
        if self.method == 'fgsm':
            return self._fgsm_attack(audio, model, epsilon, target_class)
        else:
            raise ValueError(f"Unknown attack method: {self.method}")
    
    def _fgsm_attack(self, audio: torch.Tensor, model, epsilon: float,
                     target_class: int) -> torch.Tensor:
        """
        Fast Gradient Sign Method (FGSM)
        
        Mathematical formulation:
        x_adv = x + ε * sign(∇_x L(θ, x, y))
        
        where:
        - x is the input
        - ε is the perturbation magnitude
        - L is the loss function
        - y is the true label
        """
        # Ensure audio requires gradient
        audio_var = audio.clone().detach().requires_grad_(True)
        
        # Extract features for LID model
        features = self._extract_lid_features(audio_var)
        
        # Forward pass
        model.eval()
        output = model(features)
        
        # Handle tuple output (probs, labels)
        if isinstance(output, tuple):
            probs, _ = output
            output = probs
        
        # Flatten output if needed: [batch, time, classes] -> [batch, classes]
        if output.dim() > 2:
            output = output.mean(dim=1)  # Average over time dimension
        
        # Compute loss
        criterion = nn.CrossEntropyLoss()
        target = torch.tensor([target_class], dtype=torch.long).to(self.device)
        loss = criterion(output, target)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Get gradient sign
        grad_sign = audio_var.grad.sign()
        
        # Generate adversarial example
        adv_audio = audio + epsilon * grad_sign
        
        # Clip to valid audio range
        adv_audio = torch.clamp(adv_audio, -1.0, 1.0)
        
        return adv_audio.detach()
    
    def _get_lid_prediction(self, audio: torch.Tensor, model) -> int:
        """Get LID prediction for audio"""
        model.eval()
        
        with torch.no_grad():
            features = self._extract_lid_features(audio)
            output = model(features)
            
            # Handle tuple output (probs, labels)
            if isinstance(output, tuple):
                probs, labels = output
                pred = labels[0, 0] if labels.dim() > 1 else labels[0]
            else:
                pred = torch.argmax(output, dim=-1)
        
        return pred.item()
    
    def _extract_lid_features(self, audio: torch.Tensor) -> torch.Tensor:
        """Extract features for LID model"""
        import torchaudio
        
        # Compute mel spectrogram
        mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=22050,
            n_fft=512,
            hop_length=160,
            n_mels=80
        ).to(self.device)
        
        mel_spec = mel_transform(audio.unsqueeze(0))
        
        # Convert to log scale
        log_mel = torch.log(mel_spec + 1e-8)
        
        return log_mel
    
    def _compute_snr(self, original: torch.Tensor, adversarial: torch.Tensor) -> float:
        """
        Compute Signal-to-Noise Ratio
        
        SNR = 10 * log10(P_signal / P_noise)
        
        Args:
            original: Original audio
            adversarial: Adversarial audio
            
        Returns:
            SNR in dB
        """
        # Compute noise
        noise = adversarial - original
        
        # Compute powers
        signal_power = torch.mean(original ** 2)
        noise_power = torch.mean(noise ** 2)
        
        # Compute SNR
        snr = 10 * torch.log10(signal_power / (noise_power + 1e-10))
        
        return snr.item()
