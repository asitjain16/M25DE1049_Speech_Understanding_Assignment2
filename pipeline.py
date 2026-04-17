#!/usr/bin/env python3
"""
Main pipeline for Code-Switched Speech Processing Assignment
Integrates all four parts: STT, Translation, TTS, and Adversarial Testing
"""

import argparse
import yaml
import logging
from pathlib import Path
import torch
import time

from src.part1_stt.transcription_pipeline import TranscriptionPipeline
from src.part2_translation.translation_pipeline import TranslationPipeline
from src.part3_tts.synthesis_pipeline import SynthesisPipeline
from src.part4_adversarial.adversarial_pipeline import AdversarialPipeline
from src.utils.audio_utils import load_audio, save_audio
from src.utils.metrics import compute_all_metrics

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SpeechProcessingPipeline:
    """Complete pipeline for code-switched speech processing"""
    
    def __init__(self, config_path: str):
        """Initialize pipeline with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize sub-pipelines
        self.stt_pipeline = None
        self.translation_pipeline = None
        self.tts_pipeline = None
        self.adversarial_pipeline = None
        
        # Create output directories
        self._setup_directories()
    
    def _setup_directories(self):
        """Create necessary directories"""
        dirs = [
            'data/raw', 'data/processed', 'data/reference',
            'models/lid', 'models/stt', 'models/tts', 'models/antispoofing',
            'outputs/transcripts', 'outputs/audio', 'outputs/metrics',
            'report'
        ]
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def run_part1_stt(self, audio_path: str) -> dict:
        """
        Part 1: Robust Code-Switched Transcription
        
        Returns:
            dict: Contains transcript, language labels, and timestamps
        """
        logger.info("PART 1: Speech-to-Text with Language Identification")
        
        if self.stt_pipeline is None:
            self.stt_pipeline = TranscriptionPipeline(self.config['stt'], self.device)
        
        logger.info("Denoising audio...")
        denoised_audio = self.stt_pipeline.denoise_audio(audio_path)
        
        logger.info("Running frame-level Language ID...")
        lid_results = self.stt_pipeline.identify_languages(denoised_audio)
        
        logger.info("Transcribing with constrained decoding...")
        transcript = self.stt_pipeline.transcribe_with_constraints(
            denoised_audio, 
            lid_results
        )
        
        output = {
            'transcript': transcript,
            'lid_results': lid_results,
            'denoised_audio_path': 'outputs/audio/denoised.wav'
        }
        
        save_audio(denoised_audio, output['denoised_audio_path'], 
                   self.config['audio']['sample_rate'])
        
        with open('outputs/transcripts/part1_transcript.txt', 'w', encoding='utf-8') as f:
            f.write(transcript['text'])
        
        logger.info(f"Part 1 complete. WER: {transcript.get('wer', 'N/A')}")
        return output
    
    def run_part2_translation(self, transcript: dict) -> dict:
        """
        Part 2: Phonetic Mapping & Translation
        
        Returns:
            dict: Contains IPA representation and LRL translation
        """
        logger.info("PART 2: Phonetic Mapping & Translation")
        
        if self.translation_pipeline is None:
            self.translation_pipeline = TranslationPipeline(
                self.config['translation'], 
                self.device
            )
        
        logger.info("Converting to IPA representation...")
        ipa_text = self.translation_pipeline.convert_to_ipa(
            transcript['text'],
            transcript.get('language_segments', [])
        )
        
        logger.info("Translating to target LRL...")
        lrl_translation = self.translation_pipeline.translate_to_lrl(
            ipa_text,
            target_lang=self.config['translation']['target_lrl']
        )
        
        output = {
            'ipa_text': ipa_text,
            'lrl_translation': lrl_translation
        }
        
        with open('outputs/transcripts/part2_ipa.txt', 'w', encoding='utf-8') as f:
            f.write(ipa_text)
        
        with open('outputs/transcripts/part2_lrl_translation.txt', 'w', encoding='utf-8') as f:
            f.write(lrl_translation)
        
        logger.info("Part 2 complete.")
        return output
    
    def run_part3_tts(self, lrl_text: str, original_audio_path: str) -> dict:
        """
        Part 3: Zero-Shot Cross-Lingual Voice Cloning
        
        Returns:
            dict: Contains synthesized audio path and metrics
        """
        logger.info("PART 3: Zero-Shot Voice Cloning with Prosody Transfer")
        
        if self.tts_pipeline is None:
            self.tts_pipeline = SynthesisPipeline(self.config['tts'], self.device)
        
        logger.info("Extracting speaker embedding from reference...")
        reference_path = self.config['tts']['reference']['path']
        speaker_embedding = self.tts_pipeline.extract_speaker_embedding(reference_path)
        
        logger.info("Extracting and warping prosody features...")
        prosody_features = self.tts_pipeline.extract_and_warp_prosody(
            original_audio_path,
            reference_path
        )
        
        logger.info("Synthesizing LRL speech...")
        synthesized_audio = self.tts_pipeline.synthesize(
            lrl_text,
            speaker_embedding,
            prosody_features
        )
        
        output_path = 'outputs/audio/output_LRL_cloned.wav'
        save_audio(synthesized_audio, output_path, 
                   self.config['tts']['output']['sample_rate'])
        
        mcd = self.tts_pipeline.compute_mcd(synthesized_audio, reference_path)
        
        output = {
            'synthesized_audio_path': output_path,
            'mcd': mcd,
            'prosody_features': prosody_features
        }
        
        logger.info(f"Part 3 complete. MCD: {mcd:.2f}")
        return output
    
    def run_part4_adversarial(self, audio_paths: dict) -> dict:
        """
        Part 4: Adversarial Robustness & Spoofing Detection
        
        Returns:
            dict: Contains EER, adversarial results, and metrics
        """
        logger.info("PART 4: Adversarial Robustness & Anti-Spoofing")
        
        if self.adversarial_pipeline is None:
            self.adversarial_pipeline = AdversarialPipeline(
                self.config['adversarial'],
                self.device
            )
        
        logger.info("Training and evaluating anti-spoofing classifier...")
        antispoofing_results = self.adversarial_pipeline.train_antispoofing(
            real_audio=audio_paths['reference'],
            fake_audio=audio_paths['synthesized']
        )
        
        logger.info("Testing adversarial robustness...")
        
        # Initialize STT pipeline if needed (for LID model)
        if self.stt_pipeline is None:
            from src.part1_stt.transcription_pipeline import TranscriptionPipeline
            self.stt_pipeline = TranscriptionPipeline(self.config['stt'], self.device)
        
        adversarial_results = self.adversarial_pipeline.test_adversarial_robustness(
            audio_paths['original'],
            self.stt_pipeline.lid_model
        )
        
        output = {
            'eer': antispoofing_results['eer'],
            'min_epsilon': adversarial_results['min_epsilon'],
            'snr': adversarial_results['snr'],
            'confusion_matrix': antispoofing_results['confusion_matrix']
        }
        
        logger.info(f"Part 4 complete. EER: {output['eer']:.4f}, Min epsilon: {output['min_epsilon']:.6f}")
        
        return output
    
    def run_full_pipeline(self, audio_path: str):
        """Run complete pipeline from audio to synthesized LRL speech"""
        logger.info("Starting full pipeline execution...")
        
        # Part 1: STT
        stt_results = self.run_part1_stt(audio_path)
        
        # Part 2: Translation
        translation_results = self.run_part2_translation(stt_results['transcript'])
        
        # Part 3: TTS
        tts_results = self.run_part3_tts(
            translation_results['lrl_translation'],
            stt_results['denoised_audio_path']
        )
        
        # Part 4: Adversarial
        audio_paths = {
            'original': audio_path,
            'reference': self.config['tts']['reference']['path'],
            'synthesized': tts_results['synthesized_audio_path']
        }
        adversarial_results = self.run_part4_adversarial(audio_paths)
        
        # Compile final results
        final_results = {
            'stt': stt_results,
            'translation': translation_results,
            'tts': tts_results,
            'adversarial': adversarial_results
        }
        
        self._save_final_report(final_results)
        
        logger.info("Pipeline execution complete.")
        
        return final_results
    
    def _save_final_report(self, results: dict):
        """Save comprehensive evaluation report"""
        import json
        
        report_path = 'outputs/metrics/final_report.json'
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Final report saved to {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Code-Switched Speech Processing Pipeline'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--audio', 
        type=str,
        default='data/processed/original_segment.wav',
        help='Path to input audio file'
    )
    parser.add_argument(
        '--part',
        type=int,
        choices=[1, 2, 3, 4],
        help='Run specific part only (1-4)'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = SpeechProcessingPipeline(args.config)
    
    # Run requested part or full pipeline
    if args.part == 1:
        pipeline.run_part1_stt(args.audio)
    elif args.part == 2:
        # Load previous results
        with open('outputs/transcripts/part1_transcript.txt', 'r') as f:
            transcript = {'text': f.read()}
        pipeline.run_part2_translation(transcript)
    elif args.part == 3:
        with open('outputs/transcripts/part2_lrl_translation.txt', 'r') as f:
            lrl_text = f.read()
        pipeline.run_part3_tts(lrl_text, args.audio)
    elif args.part == 4:
        audio_paths = {
            'original': args.audio,
            'reference': pipeline.config['tts']['reference']['path'],
            'synthesized': 'outputs/audio/output_LRL_cloned.wav'
        }
        pipeline.run_part4_adversarial(audio_paths)
    else:
        # Run full pipeline
        pipeline.run_full_pipeline(args.audio)


if __name__ == '__main__':
    main()
