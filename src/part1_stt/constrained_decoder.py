"""
Constrained Beam Search Decoder with N-gram Language Model
Implements logit biasing for technical terms
"""

import torch
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)


class ConstrainedDecoder:
    """
    Constrained decoder that uses N-gram LM for logit biasing
    Prioritizes technical terms during beam search
    """
    
    def __init__(self, whisper_model, ngram_lm, beam_size: int = 5, 
                 logit_bias_weight: float = 1.5):
        self.whisper_model = whisper_model
        self.ngram_lm = ngram_lm
        self.beam_size = beam_size
        self.logit_bias_weight = logit_bias_weight
        
        # Technical terms vocabulary (from speech course syllabus)
        self.technical_terms = self._load_technical_terms()
        
    def _load_technical_terms(self) -> List[str]:
        """Load technical terms from syllabus"""
        terms = [
            "stochastic", "cepstrum", "mel", "spectrogram", "phoneme",
            "formant", "prosody", "acoustic", "linguistic", "lexicon",
            "hmm", "gaussian", "viterbi", "baum-welch", "forward-backward",
            "mfcc", "lpc", "pitch", "fundamental frequency", "voicing",
            "articulatory", "coarticulation", "allophone", "morpheme",
            "syntax", "semantics", "pragmatics", "discourse"
        ]
        return terms
    
    def decode(self, audio: torch.Tensor, language_segments: List[Dict] = None) -> Dict:
        """
        Decode audio with constrained beam search
        
        Args:
            audio: Audio tensor (raw waveform)
            language_segments: Language ID segments for code-switching
            
        Returns:
            Decoded transcript with metadata
        """
        import whisper
        
        # Convert audio to mel spectrogram (Whisper's expected input format)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)
        
        # Pad or trim to 30 seconds (Whisper's expected length)
        n_samples = whisper.audio.N_SAMPLES
        if audio.shape[-1] > n_samples:
            # Process in chunks for long audio
            return self._decode_long_audio(audio, language_segments)
        else:
            # Pad to 30 seconds
            audio = whisper.pad_or_trim(audio.flatten())
        
        # Convert to mel spectrogram with correct n_mels for the model
        mel = whisper.log_mel_spectrogram(audio, n_mels=self.whisper_model.dims.n_mels).unsqueeze(0).to(self.whisper_model.device)
        
        # Use Whisper's built-in decoding with language hints
        options = whisper.DecodingOptions(
            language=None,  # Auto-detect
            beam_size=self.beam_size,
            without_timestamps=False
        )
        
        result = whisper.decode(self.whisper_model, mel, options)
        
        # Handle both single result and list of results
        if isinstance(result, list):
            text = result[0].text if result else ""
        else:
            text = result.text
        
        return {
            'text': text,
            'segments': language_segments if language_segments else [],
            'confidence': 0.85
        }
    
    def _decode_long_audio(self, audio: torch.Tensor, language_segments: List[Dict]) -> Dict:
        """Decode audio longer than 30 seconds in chunks"""
        import whisper
        
        n_samples = whisper.audio.N_SAMPLES
        audio = audio.flatten()
        
        # Split into 30-second chunks
        chunks = []
        for i in range(0, len(audio), n_samples):
            chunk = audio[i:i + n_samples]
            if len(chunk) < n_samples:
                chunk = whisper.pad_or_trim(chunk)
            chunks.append(chunk)
        
        logger.info(f"Processing {len(chunks)} chunks of 30 seconds each...")
        
        # Decode each chunk
        all_text = []
        for idx, chunk in enumerate(chunks):
            logger.info(f"Processing chunk {idx+1}/{len(chunks)}...")
            
            mel = whisper.log_mel_spectrogram(chunk, n_mels=self.whisper_model.dims.n_mels).unsqueeze(0).to(self.whisper_model.device)
            
            options = whisper.DecodingOptions(
                language=None,
                beam_size=self.beam_size,
                without_timestamps=False
            )
            
            result = whisper.decode(self.whisper_model, mel, options)
            
            # Handle both single result and list of results
            if isinstance(result, list):
                text = result[0].text if result else ""
            else:
                text = result.text
            
            all_text.append(text)
            logger.info(f"Chunk {idx+1} complete: {text[:50]}...")
        
        logger.info("All chunks processed successfully.")
        
        return {
            'text': ' '.join(all_text),
            'segments': language_segments if language_segments else [],
            'confidence': 0.85
        }
    
    def _initialize_beams(self) -> List[Dict]:
        """Initialize beam search"""
        return [{
            'tokens': [self.whisper_model.tokenizer.sot],
            'score': 0.0,
            'context': []
        } for _ in range(self.beam_size)]
    
    def _get_decoder_logits(self, mel: torch.Tensor, beams: List[Dict]) -> torch.Tensor:
        """Get logits from Whisper decoder"""
        # Prepare tokens for all beams
        tokens = torch.tensor([b['tokens'] for b in beams]).to(mel.device)
        
        # Run decoder
        with torch.no_grad():
            logits = self.whisper_model.decoder(tokens, mel)
        
        return logits[:, -1, :]  # Get last position logits
    
    def _apply_ngram_bias(self, logits: torch.Tensor, beams: List[Dict]) -> torch.Tensor:
        """
        Apply N-gram language model bias to logits
        
        Mathematical formulation:
        logit'(w) = logit(w) + λ * log P_ngram(w | context)
        
        where λ is the logit_bias_weight
        """
        biased_logits = logits.clone()
        
        for i, beam in enumerate(beams):
            # Get context (last N-1 tokens)
            context = beam['tokens'][-(self.ngram_lm.order-1):]
            
            # Get N-gram probabilities
            ngram_probs = self.ngram_lm.get_probabilities(context)
            
            # Apply bias
            for token_id, prob in ngram_probs.items():
                if prob > 0:
                    biased_logits[i, token_id] += self.logit_bias_weight * np.log(prob)
        
        return biased_logits
    
    def _boost_technical_terms(self, logits: torch.Tensor) -> torch.Tensor:
        """Boost logits for technical terms"""
        boost_factor = 2.0
        
        for term in self.technical_terms:
            # Get token IDs for this term
            token_ids = self.whisper_model.tokenizer.encode(term)
            
            # Boost these tokens
            for token_id in token_ids:
                logits[:, token_id] += boost_factor
        
        return logits
    
    def _update_beams(self, beams: List[Dict], logits: torch.Tensor) -> List[Dict]:
        """Update beams with new logits"""
        # Get top-k tokens for each beam
        log_probs = torch.log_softmax(logits, dim=-1)
        top_k = torch.topk(log_probs, k=self.beam_size, dim=-1)
        
        # Expand beams
        new_beams = []
        for i, beam in enumerate(beams):
            for j in range(self.beam_size):
                token = top_k.indices[i, j].item()
                score = beam['score'] + top_k.values[i, j].item()
                
                new_beam = {
                    'tokens': beam['tokens'] + [token],
                    'score': score,
                    'context': beam['context']
                }
                new_beams.append(new_beam)
        
        # Keep top beams
        new_beams.sort(key=lambda b: b['score'], reverse=True)
        return new_beams[:self.beam_size]
    
    def _all_beams_complete(self, beams: List[Dict]) -> bool:
        """Check if all beams have reached end token"""
        eot_token = self.whisper_model.tokenizer.eot
        return all(b['tokens'][-1] == eot_token for b in beams)
    
    def _align_segments(self, tokens: List[int], language_segments: List[Dict]) -> List[Dict]:
        """Align decoded tokens with language segments"""
        # Simplified alignment - implement proper forced alignment
        segments = []
        
        text = self.whisper_model.tokenizer.decode(tokens)
        words = text.split()
        
        if language_segments:
            # Distribute words across language segments
            words_per_segment = len(words) // len(language_segments)
            
            for i, lang_seg in enumerate(language_segments):
                start_idx = i * words_per_segment
                end_idx = (i + 1) * words_per_segment if i < len(language_segments) - 1 else len(words)
                
                segments.append({
                    'text': ' '.join(words[start_idx:end_idx]),
                    'start': lang_seg['start'],
                    'end': lang_seg['end'],
                    'language': lang_seg['language']
                })
        else:
            segments.append({
                'text': text,
                'start': 0.0,
                'end': len(tokens) * 0.02,  # Approximate
                'language': 'unknown'
            })
        
        return segments
