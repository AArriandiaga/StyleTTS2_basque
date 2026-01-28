import subprocess
import logging
import string
from pathlib import Path
from collections import OrderedDict
from nltk.tokenize import TweetTokenizer
from typing import List, Dict, Optional
import re

# Constants
SUPPORTED_LANGUAGES = {'eu', 'es'}
SUPPORTED_SYMBOLS = {'sampa', 'ipa'}
SAMPA_TO_IPA = OrderedDict([
    ("p", "p"), ("b", "b"), ("t", "t"), ("c", "c"), ("d", "d"),
    ("k", "k"), ("g", "ɡ"), ("tS", "tʃ"), ("ts", "ts"), ("ts`", "tʂ"),
    ("gj", "ɟ"), ("jj", "ʝ"), ("f", "f"), ("B", "β"), ("T", "θ"),
    ("D", "ð"), ("s", "s"), ("s`", "ʂ"), ("S", "ʃ"), ("x", "x"),
    ("G", "ɣ"), ("m", "m"), ("n", "n"), ("J", "ɲ"), ("l", "l"),
    ("L", "ʎ"), ("r", "ɾ"), ("rr", "r"), ("j", "j"), ("w", "w"),
    ("i", "i"), ("'i", "'i"), ("e", "e"), ("'e", "'e"), ("a", "a"),
    ("'a", "'a"), ("o", "o"), ("'o", "'o"), ("u", "u"), ("'u", "'u"),
    ("y", "y"), ("Z", "ʒ"), ("h", "h"), ("ph", "pʰ"), ("kh", "kʰ"),
    ("th", "tʰ")
])

MULTICHAR_TO_SINGLECHAR = {
    "tʃ": "C",
    "ts": "V",
    "tʂ": "P",
    "'i": "I",
    "'e": "E",
    "'a": "A",
    "'o": "O",
    "'u": "U",
    "pʰ": "H",
    "kʰ": "K",
    "tʰ": "T"
}

class PhonemizerError(Exception):
    """Custom exception for Phonemizer errors."""
    pass

class Phonemizer:
    def __init__(self, language: str = "eu", symbol: str = "sampa", 
                path_modulo1y2: str = "modulo1y2/modulo1y2", 
                path_dicts: str = "dict") -> None:
        """Initialize the Phonemizer with the given language and symbol."""
        if language not in SUPPORTED_LANGUAGES:
            raise PhonemizerError(f"Unsupported language: {language}")
        if symbol not in SUPPORTED_SYMBOLS:
            raise PhonemizerError(f"Unsupported symbol type: {symbol}")

        self.language = language
        self.symbol = symbol
        self.path_modulo1y2 = Path(path_modulo1y2)
        self.path_dicts = Path(path_dicts)
        self.logger = logging.getLogger(__name__)
        
        # Initialize SAMPA to IPA dictionary
        self._sampa_to_ipa_dict = SAMPA_TO_IPA
        
        # Initialize word splitter regex
        self._word_splitter = re.compile(r'\w+|[^\w\s]', re.UNICODE)
        
        self._validate_paths()

    def normalize(self, text: str) -> str:
        """Normalize the given text using an external command."""
        try:
            command = self._build_normalization_command()
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='ISO-8859-15',
                shell=True
            )
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                # Filter out the SetDur warning from the error message
                filtered_stderr = '\n'.join(line for line in stderr.split('\n') 
                                          if 'Warning: argument not used SetDur' not in line)
                if filtered_stderr.strip():  # Only raise error if there are other errors
                    error_msg = f"Normalization failed: {filtered_stderr}"
                    self.logger.error(error_msg)
                    raise PhonemizerError(error_msg)
            
            return stdout.strip()
            
        except Exception as e:
            error_msg = f"Error during normalization: {str(e)}"
            self.logger.error(error_msg)
            return text

    def getPhonemes(self, text: str, use_single_char: bool = False) -> str:
        """Extract phonemes from the given text.
        
        Args:
            text (str): The input text to convert to phonemes
            use_single_char (bool): If True, converts multi-character IPA phonemes to single characters
                                and joins them without spaces. If False, keeps phonemes separated by spaces.
                                Only applies when symbol="ipa". Defaults to False.
        
        Returns:
            str: The phoneme sequence with words separated by " | "
        """
        try:
            # Pre-process text to handle dots consistently
            # Replace multiple dots with a single dot to avoid issues with ellipsis
            text = re.sub(r'\.{2,}', '.', text)
            
            command = self._build_phoneme_extraction_command()
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='ISO-8859-15',
                shell=True
            )
            
            stdout, stderr = process.communicate(input=text)
            
            if process.returncode != 0:
                error_msg = f"Phoneme extraction failed: {stderr}"
                self.logger.error(error_msg)
                raise PhonemizerError(error_msg)
            
            # Handle newlines in the raw phonemes output
            # Replace newlines with underscores, similar to how other punctuation is handled
            stdout = stdout.replace('\n', ' | _ | ')
            
            # Get words and punctuation from normalized text
            words = self._word_splitter.findall(text)
            
            # Split into words and handle each separately
            word_phonemes = stdout.split(" | ")
            result_phonemes = []
            
            # Clean and prepare phoneme sequences
            cleaned_phonemes = []
            for phoneme_seq in word_phonemes:
                if not phoneme_seq.strip():
                    continue
                # Remove underscores and clean up the sequence
                if phoneme_seq.strip() == "_":
                    continue
                cleaned_phonemes.append(phoneme_seq.strip())
            
            # Count non-punctuation words and punctuation marks separately
            non_punct_words = [w for w in words if w not in string.punctuation]
            punct_marks = [w for w in words if w in string.punctuation]
            
            # Ensure we have enough phonemes for all non-punctuation words
            if len(cleaned_phonemes) < len(non_punct_words):
                # If not, duplicate the last phoneme
                while len(cleaned_phonemes) < len(non_punct_words):
                    if cleaned_phonemes:
                        cleaned_phonemes.append(cleaned_phonemes[-1])
                    else:
                        # If no phonemes at all, add a placeholder
                        cleaned_phonemes.append("a")
            
            # Process words and phonemes together
            phoneme_idx = 0
            word_idx = 0
            
            while word_idx < len(words):
                word = words[word_idx]
                
                if word in string.punctuation:
                    # Add punctuation mark directly
                    result_phonemes.append(word)
                    word_idx += 1
                else:
                    # Process regular word
                    if phoneme_idx < len(cleaned_phonemes):
                        phonemes = cleaned_phonemes[phoneme_idx].split()
                        if self.symbol == "sampa":
                            # For SAMPA, join and remove hyphens
                            processed_phonemes = " ".join(p for p in phonemes if p != "-")
                        else:
                            # For IPA, convert and remove hyphens
                            ipa_phonemes = [self._sampa_to_ipa_dict.get(p, p) for p in phonemes if p != "-"]
                            processed_phonemes = " ".join(ipa_phonemes)
                            if use_single_char:
                                processed_phonemes = self._transform_multichar_phonemes(processed_phonemes)
                                # Join phonemes when use_single_char is True
                                processed_phonemes = processed_phonemes.replace(" ", "")
                        
                        result_phonemes.append(processed_phonemes)
                        phoneme_idx += 1
                        word_idx += 1
                    else:
                        # If we run out of phonemes but still have words, skip the word
                        word_idx += 1
            
            # If we have more phonemes than words, add them as is
            while phoneme_idx < len(cleaned_phonemes):
                phonemes = cleaned_phonemes[phoneme_idx].split()
                if self.symbol == "sampa":
                    processed_phonemes = " ".join(p for p in phonemes if p != "-")
                else:
                    ipa_phonemes = [self._sampa_to_ipa_dict.get(p, p) for p in phonemes if p != "-"]
                    processed_phonemes = " ".join(ipa_phonemes)
                    if use_single_char:
                        processed_phonemes = self._transform_multichar_phonemes(processed_phonemes)
                        # Join phonemes when use_single_char is True
                        processed_phonemes = processed_phonemes.replace(" ", "")
                
                result_phonemes.append(processed_phonemes)
                phoneme_idx += 1
            
            return " | ".join(result_phonemes)
            
        except Exception as e:
            error_msg = f"Error in phoneme extraction: {str(e)}"
            self.logger.error(error_msg)
            return ""

    def _build_normalization_command(self) -> str:
        """Build the command string for normalization."""
        modulo_path = self._get_file_path() / self.path_modulo1y2
        dict_path = self._get_file_path() / self.path_dicts
        dict_file = f"{self.language}_dicc"
        return f'{modulo_path} -TxtMode=Word -Lang={self.language} -HDic={dict_path/dict_file}'

    def _build_phoneme_extraction_command(self) -> str:
        """Build the command string for phoneme extraction."""
        modulo_path = self._get_file_path() / self.path_modulo1y2
        dict_path = self._get_file_path() / self.path_dicts
        dict_file = f"{self.language}_dicc"
        return f'{modulo_path} -Lang={self.language} -HDic={dict_path/dict_file}'

    def _get_file_path(self) -> Path:
        return Path(__file__).parent

    def _validate_paths(self) -> None:
        """Validate paths with enhanced error reporting."""
        try:
            if not self.path_modulo1y2.exists():
                raise PhonemizerError(f"Modulo1y2 executable not found at: {self.path_modulo1y2}")
            if not self.path_dicts.exists():
                raise PhonemizerError(f"Dictionary directory not found at: {self.path_dicts}")
            
            # Check for both possible dictionary files
            dict_file = self.path_dicts / f"{self.language}_dicc"
            if not dict_file.exists():
                # Try with .dic extension as fallback
                dict_file_alt = self.path_dicts / f"{self.language}_dicc.dic"
                if not dict_file_alt.exists():
                    raise PhonemizerError(f"Dictionary file not found at either {dict_file} or {dict_file_alt}")
                
        except Exception as e:
            self.logger.error(f"Path validation error: {str(e)}")
            raise 

    def _transform_multichar_phonemes(self, phoneme_sequence: str) -> str:
        """
        Transform multicharacter IPA phonemes to single characters using the MULTICHAR_TO_SINGLECHAR mapping.
        
        Args:
            phoneme_sequence (str): A string containing phonemes separated by spaces
            
        Returns:
            str: The transformed phoneme sequence with multicharacter phonemes replaced by single characters
        """
        # Split the sequence into individual phonemes
        phonemes = phoneme_sequence.split()
        transformed_phonemes = []
        
        for phoneme in phonemes:
            # Check if the phoneme exists in our mapping
            if phoneme in MULTICHAR_TO_SINGLECHAR:
                transformed_phonemes.append(MULTICHAR_TO_SINGLECHAR[phoneme])
            else:
                transformed_phonemes.append(phoneme)
        
        return " ".join(transformed_phonemes)

    