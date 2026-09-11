"""
Grammar-aware Sentence Generator for Sign Language Translation.
Combines detected signs into grammatically correct English sentences.
"""

import time
from src.config import WORD_CATEGORIES, SENTENCE_TIMEOUT


class SentenceGenerator:
    """
    Accumulates detected sign words and forms grammatically correct sentences.
    
    Features:
        - Word categorization (subject, verb, object, greeting, modifier, affirmation)
        - Grammar-aware sentence construction (SVO order)
        - Timeout-based sentence reset
        - Duplicate word prevention
        - Sentence history tracking
    """
    
    def __init__(self, word_timeout=SENTENCE_TIMEOUT):
        self.categories = WORD_CATEGORIES
        self.sentence_buffer = []
        self.last_detection_time = time.time()
        self.word_timeout = word_timeout
        self.sentence_history = []
        self.current_sentence = ""
    
    def categorize_word(self, word):
        """Return the grammatical category of a word."""
        word_lower = word.lower()
        for category, words in self.categories.items():
            if word_lower in words:
                return category
        return 'unknown'
    
    def add_word(self, word):
        """
        Add a detected sign word to the sentence buffer.
        
        Returns the current formed sentence.
        """
        current_time = time.time()
        word_lower = word.lower()
        
        # Reset buffer if too much time has passed
        if current_time - self.last_detection_time > self.word_timeout:
            if self.current_sentence:
                self.sentence_history.append(self.current_sentence)
            self.sentence_buffer = []
        
        self.last_detection_time = current_time
        
        # Don't add consecutive duplicates
        if self.sentence_buffer and self.sentence_buffer[-1] == word_lower:
            return self.current_sentence
        
        self.sentence_buffer.append(word_lower)
        self.current_sentence = self._form_sentence()
        return self.current_sentence
    
    def _form_sentence(self):
        """
        Form a grammatically correct sentence from the buffer.
        
        Handles:
            - Single greetings/affirmations with proper punctuation
            - SVO (Subject-Verb-Object) sentence construction
            - Modifier placement (please, sorry)
            - Mixed word types
        """
        if not self.sentence_buffer:
            return ""
        
        # Single word handling
        if len(self.sentence_buffer) == 1:
            word = self.sentence_buffer[0]
            category = self.categorize_word(word)
            
            if category == 'greeting':
                return word.capitalize() + "!"
            elif category == 'affirmation':
                return word.capitalize() + "."
            elif category == 'modifier':
                return word.capitalize() + "."
            else:
                return word.capitalize()
        
        # Multi-word sentence construction
        subject = None
        verb = None
        obj = None
        greetings = []
        modifiers = []
        affirmations = []
        extra_words = []
        
        for word in self.sentence_buffer:
            category = self.categorize_word(word)
            
            if category == 'subject' and subject is None:
                subject = word
            elif category == 'verb' and verb is None:
                verb = word
            elif category == 'object' and obj is None:
                obj = word
            elif category == 'greeting':
                greetings.append(word)
            elif category == 'modifier':
                modifiers.append(word)
            elif category == 'affirmation':
                affirmations.append(word)
            elif category == 'subject' and subject is not None:
                extra_words.append(word)
            elif category == 'verb' and verb is not None:
                extra_words.append(word)
            elif category == 'object' and obj is not None:
                extra_words.append(word)
        
        # Build sentence parts
        parts = []
        
        # Greetings come first
        for g in greetings:
            parts.append(g.capitalize() + "!")
        
        # Affirmations
        for a in affirmations:
            parts.append(a.capitalize() + ".")
        
        # Build SVO clause
        svo_parts = []
        
        # "Sorry" goes before the subject
        if 'sorry' in modifiers:
            svo_parts.append('Sorry')
            modifiers.remove('sorry')
        
        if subject:
            # Handle pronoun capitalization
            if subject == 'i':
                svo_parts.append('I')
            else:
                svo_parts.append(subject.capitalize() if not svo_parts else subject)
        
        if verb:
            # Conjugate for third person if needed (future improvement)
            svo_parts.append(verb)
        
        if obj:
            svo_parts.append(obj)
        
        # "Please" goes at the end
        if 'please' in modifiers:
            svo_parts.append('please')
            modifiers.remove('please')
        
        # Any remaining modifiers
        for m in modifiers:
            svo_parts.append(m)
        
        if svo_parts:
            svo_sentence = ' '.join(svo_parts)
            # Add period if we have a complete-ish sentence
            if subject and verb:
                svo_sentence += '.'
            parts.append(svo_sentence)
        
        return ' '.join(parts)
    
    def clear(self):
        """Clear the current sentence buffer."""
        if self.current_sentence:
            self.sentence_history.append(self.current_sentence)
        self.sentence_buffer = []
        self.current_sentence = ""
    
    def get_history(self):
        """Return all previously completed sentences."""
        return self.sentence_history.copy()
    
    def save_sentences(self, filepath):
        """Save all sentences (history + current) to a file."""
        all_sentences = self.sentence_history.copy()
        if self.current_sentence:
            all_sentences.append(self.current_sentence)
        
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write('\n'.join(all_sentences) + '\n')
        
        return len(all_sentences)
    
    def check_timeout(self):
        """Check if the sentence should be finalized due to timeout."""
        if (self.sentence_buffer and 
            time.time() - self.last_detection_time > self.word_timeout):
            if self.current_sentence:
                self.sentence_history.append(self.current_sentence)
            self.sentence_buffer = []
            self.current_sentence = ""
            return True
        return False
