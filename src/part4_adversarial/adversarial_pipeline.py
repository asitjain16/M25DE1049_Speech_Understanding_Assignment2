"""
Part 4: Adversarial Robustness & Spoofing Detection
Implements anti-spoofing classifier and adversarial attacks
"""

import torch
import numpy as np
from typing import Dict, Tuple
import logging

from .antispoofing_model import AntiSpoofingModel
from .adversarial_attack import AdversarialAttack

logger = logging.getLogger(__name__)


class AdversarialPipeline:
    """Pipeline for adversarial testing and anti-spoofing"""
    
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        
        # Task 4.1: Initialize anti-spoofing model
        self.antispoofing_model = AntiSpoofingModel(
            feature_type=config['antispoofing']['features'],
            model_type=config['antispoofing']['model'],
            device=device
        )
        
        # Task 4.2: Initialize adversarial attack
        self.adversarial_attack = AdversarialAttack(
            method=config['attack']['method'],
            epsilon_range=config['attack']['epsilon_range'],
            target_snr=config['attack']['target_snr'],
            device=device
        )
    
    def train_antispoofing(self, real_audio: str, fake_audio: str) -> Dict:
        """
        Task 4.1: Train and evaluate anti-spoofing classifier
        
        Args:
            real_audio: Path to real (bona fide) audio
            fake_audio: Path to synthesized (spoof) audio
            
        Returns:
            Dictionary with EER and confusion matrix
        """
        logger.info("Training anti-spoofing classifier...")
        
        # Prepare dataset
        train_loader, val_loader, test_loader = self.antispoofing_model.prepare_data(
            real_audio, fake_audio
        )
        
        # Train model
        self.antispoofing_model.train(train_loader, val_loader)
        
        # Evaluate on test set
        results = self.antispoofing_model.evaluate(test_loader)
        
        logger.info(f"Anti-spoofing EER: {results['eer']:.4f}")
        
        if results['eer'] >= self.config['antispoofing']['target_eer']:
            logger.warning(f"EER {results['eer']:.4f} exceeds target "
                         f"{self.config['antispoofing']['target_eer']}")
        
        return results
    
    def test_adversarial_robustness(self, audio_path: str, lid_model) -> Dict:
        """
        Task 4.2: Test adversarial robustness of LID system
        
        Args:
            audio_path: Path to audio segment
            lid_model: Language ID model to attack
            
        Returns:
            Dictionary with minimum epsilon and SNR
        """
        logger.info("Testing adversarial robustness...")
        
        # Load audio segment (5 seconds)
        import librosa
        audio, sr = librosa.load(audio_path, sr=22050, duration=5)
        
        # Find minimum perturbation to flip LID prediction
        results = self.adversarial_attack.find_minimum_perturbation(
            audio, sr, lid_model
        )
        
        logger.info(f"Minimum epsilon: {results['min_epsilon']:.6f}")
        logger.info(f"SNR: {results['snr']:.2f} dB (target: > 40 dB)")
        
        if results['snr'] < self.config['attack']['target_snr']:
            logger.warning(f"SNR {results['snr']:.2f} below target "
                         f"{self.config['attack']['target_snr']} dB")
        
        return results
