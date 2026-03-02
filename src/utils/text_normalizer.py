import re
import unicodedata
import emoji
from typing import Optional

class TextNormalizer:
    @staticmethod
    def normalize(text: Optional[str]) -> str:
        if not text:
            return ""
        
        # 1. Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # 2. Handle different line endings (Windows vs Unix)
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        # 3. Remove zero-width spaces and other invisible characters
        text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff]', '', text)
        
        # 4. Normalize multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)  # 3+ newlines -> 2 newlines
        
        # 5. Normalize spaces (but preserve paragraph breaks)
        lines = text.split('\n')
        lines = [re.sub(r' +', ' ', line.strip()) for line in lines]
        text = '\n'.join(lines)
        
        # 6. Remove spaces before punctuation
        text = re.sub(r'\s+([.,!?:;)])', r'\1', text)
        
        # 7. Add space after punctuation if missing
        text = re.sub(r'([.,!?:;)])([A-Za-z])', r'\1 \2', text)
        
        # 8. Fix common OCR errors
        text = text.replace('|', 'I')  # Pipe often misread as I
        text = text.replace('0', 'O')  # Zero often misread as O (in specific contexts)
        
        return text.strip()