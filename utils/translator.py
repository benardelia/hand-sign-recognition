import json
import os
from utils.logger_config import setup_logger

logger = setup_logger(__name__)

class GlossTranslator:
    def __init__(self, mapping_file=None):
        self.mapping = {
            "I WANT WATER": "I would like some water.",
            "I WANT EAT": "I am hungry, I want to eat.",
            "HELLO HOW YOU": "Hello, how are you?",
            "THANK YOU": "Thank you very much.",
            "NAME ME": "My name is...",
            "A B C": "Alphabet start."
        }
        
        # Load external mapping if provided
        if mapping_file and os.path.exists(mapping_file):
            try:
                with open(mapping_file, 'r') as f:
                    self.mapping.update(json.load(f))
                logger.info(f"Loaded custom translations from {mapping_file}")
            except Exception as e:
                logger.error(f"Failed to load translations: {e}")

    def translate(self, gloss_list):
        """
        Translates a list of glosses into a natural sentence.
        """
        if not gloss_list:
            return ""
        
        # Convert list to string key
        gloss_str = " ".join(gloss_list).upper()
        
        # 1. Check for exact phrase matches
        if gloss_str in self.mapping:
            return self.mapping[gloss_str]
        
        # 2. Heuristic: If no exact match, just join and fix casing
        # This is a fallback for simple word-by-word translation
        sentence = " ".join(gloss_list).capitalize()
        if not sentence.endswith(('.', '?', '!')):
            sentence += "."
            
        return sentence

# Global instance
translator = GlossTranslator()
