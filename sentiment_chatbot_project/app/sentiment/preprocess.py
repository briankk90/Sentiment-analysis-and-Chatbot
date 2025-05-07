from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

def preprocess_texts(texts, max_words=5000, max_len=100):
    """
    Preprocess texts for sentiment analysis by tokenizing and padding.
    
    Args:
        texts (list): List of text strings
        max_words (int): Maximum number of words to keep in vocabulary
        max_len (int): Maximum length of sequences
    
    Returns:
        tuple: (padded sequences, fitted tokenizer)
    """
    tokenizer = Tokenizer(num_words=max_words, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    return padded, tokenizer

def encode_labels(labels):
    """
    Encode sentiment labels to numerical values.
    
    Args:
        labels (list): List of sentiment labels ('positive', 'negative', 'neutral')
    
    Returns:
        np.ndarray: Encoded labels (0: negative, 1: neutral, 2: positive)
    """
    label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
    return np.array([label_map[label] for label in labels])