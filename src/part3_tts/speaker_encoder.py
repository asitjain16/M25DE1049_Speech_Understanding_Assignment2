"""
Speaker Encoder for extracting speaker embeddings
Implements d-vector or x-vector extraction
"""

import torch
import torch.nn as nn
import numpy as np
import librosa
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class SpeakerEncoder(nn.Module):
    """
    Speaker encoder network for extracting speaker embeddings
    Based on x-vector architecture
    """
    
    def __init__(self, input_dim: int = 40, embedding_dim: int = 256, device: torch.device = None):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # TDNN layers (Time-Delay Neural Network)
        self.tdnn1 = nn.Conv1d(input_dim, 512, kernel_size=5, dilation=1)
        self.tdnn2 = nn.Conv1d(512, 512, kernel_size=3, dilation=2)
        self.tdnn3 = nn.Conv1d(512, 512, kernel_size=3, dilation=3)
        self.tdnn4 = nn.Conv1d(512, 512, kernel_size=1, dilation=1)
        self.tdnn5 = nn.Conv1d(512, 1500, kernel_size=1, dilation=1)
        
        # Statistics pooling
        # Concatenates mean and std across time
        
        # Segment-level layers
        self.segment1 = nn.Linear(3000, 512)  # 1500 * 2 (mean + std)
        self.segment2 = nn.Linear(512, embedding_dim)
        
        # Batch normalization
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(512)
        self.bn3 = nn.BatchNorm1d(512)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(1500)
        
        self.to(self.device)
        
        # Load pretrained weights if available
        self._load_pretrained()
    
    def _load_pretrained(self):
        """Load pretrained speaker encoder weights"""
        try:
            # Try to load pretrained model
            checkpoint_path = 'models/tts/speaker_encoder.pt'
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(state_dict)
            logger.info("Loaded pretrained speaker encoder")
        except FileNotFoundError:
            logger.warning("No pretrained speaker encoder found, using random initialization")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input features (batch, features, time)
            
        Returns:
            Speaker embeddings (batch, embedding_dim)
        """
        # TDNN layers with ReLU and batch norm
        x = torch.relu(self.bn1(self.tdnn1(x)))
        x = torch.relu(self.bn2(self.tdnn2(x)))
        x = torch.relu(self.bn3(self.tdnn3(x)))
        x = torch.relu(self.bn4(self.tdnn4(x)))
        x = torch.relu(self.bn5(self.tdnn5(x)))
        
        # Statistics pooling
        mean = torch.mean(x, dim=2)
        std = torch.std(x, dim=2)
        stats = torch.cat([mean, std], dim=1)
        
        # Segment-level layers
        x = torch.relu(self.segment1(stats))
        embedding = self.segment2(x)
        
        # L2 normalization
        embedding = embedding / torch.norm(embedding, p=2, dim=1, keepdim=True)
        
        return embedding
    
    def encode(self, audio: np.ndarray, sr: int) -> torch.Tensor:
        """
        Extract speaker embedding from audio
        
        Args:
            audio: Audio array
            sr: Sample rate
            
        Returns:
            Speaker embedding tensor
        """
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(
            y=audio,
            sr=sr,
            n_mfcc=self.input_dim,
            n_fft=512,
            hop_length=160,
            win_length=400
        )
        
        # Normalize features
        mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / (np.std(mfcc, axis=1, keepdims=True) + 1e-8)
        
        # Convert to tensor
        features = torch.from_numpy(mfcc).float().unsqueeze(0).to(self.device)
        
        # Extract embedding
        with torch.no_grad():
            embedding = self.forward(features)
        
        return embedding.squeeze(0)
    
    def compute_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Compute cosine similarity between two embeddings
        
        Args:
            embedding1: First embedding
            embedding2: Second embedding
            
        Returns:
            Similarity score [0, 1]
        """
        similarity = torch.cosine_similarity(
            embedding1.unsqueeze(0),
            embedding2.unsqueeze(0)
        )
        return similarity.item()
