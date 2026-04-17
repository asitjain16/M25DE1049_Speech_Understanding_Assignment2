"""
N-gram Language Model trained on Speech Course Syllabus
Used for constrained decoding
"""

import numpy as np
from collections import defaultdict, Counter
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class NGramLanguageModel:
    """
    N-gram language model with Kneser-Ney smoothing
    Trained on technical speech course content
    """
    
    def __init__(self, order: int = 3):
        self.order = order
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = defaultdict(int)
        self.vocabulary = set()
        
    def train(self, corpus: List[str]):
        """
        Train N-gram model on corpus
        
        Args:
            corpus: List of sentences/documents
        """
        logger.info(f"Training {self.order}-gram language model...")
        
        for text in corpus:
            tokens = self._tokenize(text)
            self.vocabulary.update(tokens)
            
            # Count N-grams
            for n in range(1, self.order + 1):
                for i in range(len(tokens) - n + 1):
                    ngram = tuple(tokens[i:i+n])
                    context = ngram[:-1] if n > 1 else ()
                    word = ngram[-1]
                    
                    self.ngram_counts[context][word] += 1
                    self.context_counts[context] += 1
        
        logger.info(f"Vocabulary size: {len(self.vocabulary)}")
        logger.info(f"Unique {self.order}-grams: {len(self.ngram_counts)}")
    
    def get_probabilities(self, context: List[int]) -> Dict[int, float]:
        """
        Get probability distribution over next tokens given context
        
        Args:
            context: List of previous token IDs
            
        Returns:
            Dictionary mapping token IDs to probabilities
        """
        # Convert to tuple for lookup
        context_tuple = tuple(context[-(self.order-1):])
        
        # Get counts for this context
        word_counts = self.ngram_counts.get(context_tuple, Counter())
        total_count = self.context_counts.get(context_tuple, 0)
        
        if total_count == 0:
            # Backoff to lower order
            if len(context_tuple) > 0:
                return self.get_probabilities(list(context_tuple[1:]))
            else:
                # Uniform distribution
                return {w: 1.0 / len(self.vocabulary) for w in self.vocabulary}
        
        # Compute probabilities with Kneser-Ney smoothing
        probs = {}
        discount = 0.75  # Discount parameter
        
        for word, count in word_counts.items():
            probs[word] = max(count - discount, 0) / total_count
        
        # Add backoff probability
        lambda_weight = (discount * len(word_counts)) / total_count
        
        if len(context_tuple) > 0:
            backoff_probs = self.get_probabilities(list(context_tuple[1:]))
            for word, prob in backoff_probs.items():
                if word not in probs:
                    probs[word] = lambda_weight * prob
        
        return probs
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple tokenization"""
        return text.lower().split()
    
    def load_speech_syllabus(self, syllabus_path: str = None):
        """
        Load and train on speech course syllabus
        
        Args:
            syllabus_path: Path to syllabus file
        """
        # Default technical corpus if no syllabus provided
        default_corpus = [
            "speech recognition uses hidden markov models and gaussian mixture models",
            "the mel frequency cepstral coefficients are extracted from the audio signal",
            "acoustic models represent the relationship between audio and phonemes",
            "language models assign probabilities to sequences of words",
            "the viterbi algorithm finds the most likely state sequence",
            "forward backward algorithm computes posterior probabilities",
            "baum welch algorithm trains hidden markov model parameters",
            "fundamental frequency represents the pitch of voiced sounds",
            "formants are resonant frequencies of the vocal tract",
            "coarticulation affects phoneme realization in continuous speech",
            "prosody includes pitch rhythm and stress patterns",
            "spectral analysis decomposes signals into frequency components",
            "linear predictive coding models the vocal tract as a filter",
            "dynamic time warping aligns sequences of different lengths",
            "phonemes are the smallest units of sound in language",
            "allophonic variation depends on phonetic context",
            "articulatory features describe how sounds are produced",
            "stochastic processes model uncertainty in speech signals",
            "cepstrum is the inverse fourier transform of log spectrum",
            "lexicon maps words to their phonetic pronunciations"
        ]
        
        if syllabus_path:
            try:
                with open(syllabus_path, 'r', encoding='utf-8') as f:
                    corpus = f.readlines()
            except FileNotFoundError:
                logger.warning(f"Syllabus file not found: {syllabus_path}")
                corpus = default_corpus
        else:
            corpus = default_corpus
        
        self.train(corpus)
    
    def perplexity(self, test_corpus: List[str]) -> float:
        """
        Compute perplexity on test corpus
        
        Args:
            test_corpus: List of test sentences
            
        Returns:
            Perplexity score
        """
        log_prob_sum = 0.0
        word_count = 0
        
        for text in test_corpus:
            tokens = self._tokenize(text)
            
            for i in range(self.order - 1, len(tokens)):
                context = tokens[i-(self.order-1):i]
                word = tokens[i]
                
                probs = self.get_probabilities(context)
                prob = probs.get(word, 1e-10)
                
                log_prob_sum += np.log(prob)
                word_count += 1
        
        perplexity = np.exp(-log_prob_sum / word_count)
        return perplexity
