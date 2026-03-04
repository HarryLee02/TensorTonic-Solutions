import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id[self.pad_token] = 0
        self.word_to_id[self.unk_token] = 1
        self.word_to_id[self.bos_token] = 2
        self.word_to_id[self.eos_token] = 3

        set_text = set(" ".join(texts).split())
        
        for i, text in enumerate(set_text):
            self.word_to_id[text] = i+4
        
        self.id_to_word = {v: k for k, v in self.word_to_id.items()}

        self.vocab_size = len(self.word_to_id)
        
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        
        list_text = text.split()
        encoded_result = [self.word_to_id.get(text,1) for text in list_text] # <UNK> = 1
        return encoded_result
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        
        list_text = [self.id_to_word.get(id, self.unk_token) for id in ids]

        decoded_result = ' '.join(list_text)
        
        return decoded_result
