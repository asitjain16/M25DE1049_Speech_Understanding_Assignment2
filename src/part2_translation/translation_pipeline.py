"""
Part 2: Phonetic Mapping & Translation
IPA conversion and LRL translation
"""

import torch
from typing import List, Dict
import logging
import json

from .ipa_converter import HinglishIPAConverter
from .lrl_translator import LRLTranslator

logger = logging.getLogger(__name__)


class TranslationPipeline:
    """Pipeline for phonetic mapping and LRL translation"""
    
    def __init__(self, config: dict, device: torch.device):
        self.config = config
        self.device = device
        self.target_lrl = config['target_lrl']
        
        # Task 2.1: Initialize IPA converter
        self.ipa_converter = HinglishIPAConverter(
            use_custom_g2p=config['ipa']['use_custom_g2p'],
            rules_path=config['ipa'].get('hinglish_rules')
        )
        
        # Task 2.2: Initialize LRL translator
        self.lrl_translator = LRLTranslator(
            target_language=self.target_lrl,
            dictionary_path=config['dictionary']['path'],
            min_entries=config['dictionary']['min_entries']
        )
    
    def convert_to_ipa(self, text: str, language_segments: List[Dict] = None) -> str:
        """
        Task 2.1: Convert code-switched text to unified IPA representation
        
        Args:
            text: Input text (code-switched)
            language_segments: Language identification segments
            
        Returns:
            IPA string
        """
        logger.info("Converting to IPA representation...")
        
        # For simplicity, just convert the entire text
        # In a full implementation, you would split text by language_segments
        ipa_text = self.ipa_converter.convert(text, language='mixed')
        
        logger.info(f"IPA conversion complete. Length: {len(ipa_text)} characters")
        return ipa_text
    
    def translate_to_lrl(self, ipa_text: str, target_lang: str = None) -> str:
        """
        Task 2.2: Translate IPA/text to target Low-Resource Language
        
        Args:
            ipa_text: IPA representation or source text
            target_lang: Target LRL (overrides config)
            
        Returns:
            Translated text in LRL
        """
        if target_lang is None:
            target_lang = self.target_lrl
        
        logger.info(f"Translating to {target_lang}...")
        
        # Translate using custom dictionary and rules
        translation = self.lrl_translator.translate(ipa_text, target_lang)
        
        logger.info(f"Translation complete. Length: {len(translation)} characters")
        return translation
