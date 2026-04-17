"""
Multi-Head Language Identification Model
Frame-level classification for English and Hindi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadLID(nn.Module):
    """
    Multi-head attention-based Language ID model
    Operates at frame level with target F1 >= 0.85
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, 
                 num_languages: int, device: torch.device):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_languages = num_languages
        self.device = device
        
        # Feature extraction layers
        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Bidirectional LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, num_languages)
        )
        
        self.to(device)
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass
        
        Args:
            x: Input features (batch, features, time)
            
        Returns:
            language_probs: Probabilities for each language (batch, time, num_languages)
            language_labels: Predicted language labels (batch, time)
        """
        batch_size = x.size(0)
        
        # Convolutional feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        
        # Transpose for attention: (batch, time, features)
        x = x.transpose(1, 2)
        
        # Multi-head self-attention
        attn_output, _ = self.attention(x, x, x)
        
        # Residual connection
        x = x + attn_output
        
        # LSTM for temporal modeling
        lstm_out, _ = self.lstm(x)
        
        # Frame-level classification
        logits = self.classifier(lstm_out)
        
        # Get probabilities and predictions
        language_probs = F.softmax(logits, dim=-1)
        language_labels = torch.argmax(language_probs, dim=-1)
        
        return language_probs, language_labels
    
    def train_model(self, train_loader, val_loader, epochs: int = 50):
        """
        Train the LID model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of training epochs
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5
        )
        
        best_f1 = 0.0
        
        for epoch in range(epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            
            for batch in train_loader:
                features, labels = batch
                features = features.to(self.device)
                labels = labels.to(self.device)
                
                optimizer.zero_grad()
                
                probs, preds = self(features)
                
                # Reshape for loss computation
                probs_flat = probs.view(-1, self.num_languages)
                labels_flat = labels.view(-1)
                
                loss = criterion(probs_flat, labels_flat)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()
                
                train_loss += loss.item()
            
            # Validation phase
            val_f1 = self.evaluate(val_loader)
            scheduler.step(val_f1)
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss/len(train_loader):.4f} - "
                  f"Val F1: {val_f1:.4f}")
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save(self.state_dict(), 'models/lid/best_lid_model.pt')
                print(f"Saved best model with F1: {best_f1:.4f}")
    
    def evaluate(self, data_loader) -> float:
        """
        Evaluate model and compute F1 score
        
        Args:
            data_loader: Data loader for evaluation
            
        Returns:
            F1 score
        """
        from sklearn.metrics import f1_score
        
        self.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                features, labels = batch
                features = features.to(self.device)
                
                _, preds = self(features)
                
                all_preds.extend(preds.cpu().numpy().flatten())
                all_labels.extend(labels.numpy().flatten())
        
        f1 = f1_score(all_labels, all_preds, average='weighted')
        return f1
