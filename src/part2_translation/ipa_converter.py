"""
IPA (International Phonetic Alphabet) Converter for Hinglish
Handles code-switching between English and Hindi
"""

import re
from typing import Dict, List
import logging
import json

logger = logging.getLogger(__name__)


class HinglishIPAConverter:
    """
    Convert Hinglish text to unified IPA representation
    Handles phonological rules for code-switched speech
    """
    
    def __init__(self, use_custom_g2p: bool = True, rules_path: str = None):
        self.use_custom_g2p = use_custom_g2p
        
        # Load or create phonological rules
        if rules_path:
            self.rules = self._load_rules(rules_path)
        else:
            self.rules = self._create_default_rules()
        
        # Initialize G2P converters
        self._init_g2p_converters()
    
    def _init_g2p_converters(self):
        """Initialize grapheme-to-phoneme converters"""
        self.epi_en = None
        self.epi_hi = None
        
        try:
            from epitran import Epitran
            self.epi_en = Epitran('eng-Latn')
            self.epi_hi = Epitran('hin-Deva')
            logger.info("Epitran G2P initialized")
        except (ImportError, UnicodeDecodeError, Exception) as e:
            logger.warning(f"Epitran not available ({type(e).__name__}), using rule-based conversion")
            self.epi_en = None
            self.epi_hi = None
    
    def convert(self, text: str, language: str = 'mixed') -> str:
        """
        Convert text to IPA
        
        Args:
            text: Input text
            language: 'en', 'hi', or 'mixed'
            
        Returns:
            IPA string
        """
        if language == 'en':
            return self._convert_english(text)
        elif language == 'hi':
            return self._convert_hindi(text)
        else:
            return self._convert_mixed(text)
    
    def _convert_english(self, text: str) -> str:
        """Convert English text to IPA"""
        if self.epi_en:
            return self.epi_en.transliterate(text)
        else:
            return self._rule_based_english(text)
    
    def _convert_hindi(self, text: str) -> str:
        """Convert Hindi text to IPA"""
        if self.epi_hi:
            # Convert Romanized Hindi to Devanagari first if needed
            devanagari = self._romanized_to_devanagari(text)
            return self.epi_hi.transliterate(devanagari)
        else:
            return self._rule_based_hindi(text)
    
    def _convert_mixed(self, text: str) -> str:
        """
        Convert code-switched Hinglish to IPA
        Handles phonological phenomena at language boundaries
        """
        words = text.split()
        ipa_words = []
        
        for i, word in enumerate(words):
            # Detect language of word
            lang = self._detect_word_language(word)
            
            # Convert to IPA
            if lang == 'hi':
                ipa = self._convert_hindi(word)
            else:
                ipa = self._convert_english(word)
            
            # Apply coarticulation rules at boundaries
            if i > 0:
                prev_lang = self._detect_word_language(words[i-1])
                if prev_lang != lang:
                    ipa = self._apply_boundary_rules(ipa_words[-1], ipa, prev_lang, lang)
            
            ipa_words.append(ipa)
        
        return ' '.join(ipa_words)
    
    def _detect_word_language(self, word: str) -> str:
        """Detect if word is English or Hindi"""
        # Simple heuristic: check for Devanagari characters
        if any('\u0900' <= c <= '\u097F' for c in word):
            return 'hi'
        
        # Check against Hindi romanization patterns
        hindi_patterns = ['aap', 'hai', 'hain', 'kya', 'nahi', 'mein', 'ka', 'ki', 'ke']
        if any(pattern in word.lower() for pattern in hindi_patterns):
            return 'hi'
        
        return 'en'
    
    def _apply_boundary_rules(self, prev_ipa: str, curr_ipa: str, 
                              prev_lang: str, curr_lang: str) -> str:
        """
        Apply phonological rules at code-switch boundaries
        Handles coarticulation effects
        """
        # Example: Vowel harmony at boundaries
        if prev_ipa and curr_ipa:
            # If previous ends with vowel and current starts with vowel
            if prev_ipa[-1] in 'aeiouəɪʊ' and curr_ipa[0] in 'aeiouəɪʊ':
                # Insert glottal stop
                curr_ipa = 'ʔ' + curr_ipa
        
        return curr_ipa
    
    def _rule_based_english(self, text: str) -> str:
        """Rule-based English to IPA conversion"""
        # Simplified rule-based conversion
        ipa = text.lower()
        
        # Apply phonological rules
        for pattern, replacement in self.rules['english'].items():
            ipa = re.sub(pattern, replacement, ipa)
        
        return ipa
    
    def _rule_based_hindi(self, text: str) -> str:
        """Rule-based Hindi to IPA conversion"""
        ipa = text.lower()
        
        # Apply Hindi phonological rules
        for pattern, replacement in self.rules['hindi'].items():
            ipa = re.sub(pattern, replacement, ipa)
        
        return ipa
    
    def _romanized_to_devanagari(self, text: str) -> str:
        """Convert romanized Hindi to Devanagari script"""
        try:
            from aksharamukha import transliterate
            return transliterate.process('IAST', 'Devanagari', text)
        except ImportError:
            logger.warning("Aksharamukha not available, returning romanized text")
            return text
    
    def _create_default_rules(self) -> Dict:
        """Create default phonological rules"""
        rules = {
            'english': {
                r'th': 'θ',
                r'sh': 'ʃ',
                r'ch': 'tʃ',
                r'ng': 'ŋ',
                r'a': 'æ',
                r'e': 'ɛ',
                r'i': 'ɪ',
                r'o': 'ɔ',
                r'u': 'ʊ',
            },
            'hindi': {
                r'kh': 'kʰ',
                r'gh': 'gʱ',
                r'ch': 'tʃ',
                r'jh': 'dʒʱ',
                r'th': 'tʰ',
                r'dh': 'dʱ',
                r'ph': 'pʰ',
                r'bh': 'bʱ',
                r'a': 'ə',
                r'aa': 'aː',
                r'i': 'ɪ',
                r'ee': 'iː',
                r'u': 'ʊ',
                r'oo': 'uː',
            }
        }
        return rules
    
    def _load_rules(self, rules_path: str) -> Dict:
        """Load phonological rules from file"""
        try:
            with open(rules_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Rules file not found: {rules_path}, using defaults")
            return self._create_default_rules()
