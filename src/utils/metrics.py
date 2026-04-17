"""
Evaluation metrics for speech processing
"""

import numpy as np
from jiwer import wer, cer
from typing import Dict, List


def compute_wer(reference: str, hypothesis: str) -> float:
    """
    Compute Word Error Rate
    
    Args:
        reference: Reference transcript
        hypothesis: Hypothesis transcript
        
    Returns:
        WER value
    """
    return wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str) -> float:
    """
    Compute Character Error Rate
    
    Args:
        reference: Reference transcript
        hypothesis: Hypothesis transcript
        
    Returns:
        CER value
    """
    return cer(reference, hypothesis)


def compute_mcd(source_mfcc: np.ndarray, target_mfcc: np.ndarray) -> float:
    """
    Compute Mel-Cepstral Distortion
    
    Args:
        source_mfcc: Source MFCCs
        target_mfcc: Target MFCCs
        
    Returns:
        MCD value
    """
    # Align sequences
    min_len = min(source_mfcc.shape[1], target_mfcc.shape[1])
    source_mfcc = source_mfcc[:, :min_len]
    target_mfcc = target_mfcc[:, :min_len]
    
    # Compute MCD
    diff = source_mfcc - target_mfcc
    mcd = (10.0 / np.log(10)) * np.sqrt(2 * np.mean(diff ** 2))
    
    return mcd


def compute_lid_accuracy(predictions: List[int], ground_truth: List[int],
                        tolerance_ms: int = 200, frame_shift_ms: int = 10) -> float:
    """
    Compute LID switching accuracy with tolerance
    
    Args:
        predictions: Predicted language labels
        ground_truth: Ground truth labels
        tolerance_ms: Tolerance in milliseconds
        frame_shift_ms: Frame shift in milliseconds
        
    Returns:
        Accuracy within tolerance
    """
    tolerance_frames = tolerance_ms // frame_shift_ms
    
    correct = 0
    total = 0
    
    for i, (pred, gt) in enumerate(zip(predictions, ground_truth)):
        # Check if prediction matches within tolerance window
        start = max(0, i - tolerance_frames)
        end = min(len(ground_truth), i + tolerance_frames + 1)
        
        if pred in ground_truth[start:end]:
            correct += 1
        total += 1
    
    return correct / total if total > 0 else 0.0


def compute_all_metrics(results: Dict) -> Dict:
    """
    Compute all evaluation metrics
    
    Args:
        results: Dictionary with all pipeline results
        
    Returns:
        Dictionary with computed metrics
    """
    metrics = {
        'wer_english': results.get('stt', {}).get('wer_english', None),
        'wer_hindi': results.get('stt', {}).get('wer_hindi', None),
        'mcd': results.get('tts', {}).get('mcd', None),
        'lid_f1': results.get('stt', {}).get('lid_results', {}).get('f1_score', None),
        'eer': results.get('adversarial', {}).get('eer', None),
        'min_epsilon': results.get('adversarial', {}).get('min_epsilon', None),
        'snr': results.get('adversarial', {}).get('snr', None)
    }
    
    return metrics
