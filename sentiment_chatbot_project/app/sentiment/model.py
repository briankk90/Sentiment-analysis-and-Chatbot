from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the saved model and tokenizer
model = load_model('data/trained_model/sentiment_model.h5')
with open('data/trained_model/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

def build_model(vocab_size, embedding_dim=100, max_length=100, lstm_units=64):
    """
    Build the sentiment analysis model.
    
    Args:
        vocab_size (int): Size of the vocabulary
        embedding_dim (int): Dimension of the embedding layer
        max_length (int): Maximum length of input sequences
        lstm_units (int): Number of units in the LSTM layer
    
    Returns:
        tensorflow.keras.Model: Compiled model
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_length),
        LSTM(lstm_units, return_sequences=True),
        Dropout(0.2),
        LSTM(lstm_units // 2),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def predict_sentiment(text, max_length=100):
    """
    Predict the sentiment of the input text.
    
    Args:
        text (str): Input text to analyze
        max_length (int): Maximum length of sequences
    
    Returns:
        str: Predicted sentiment ('positive', 'negative', or 'neutral')
    """
    if tokenizer is None:
        raise ValueError("Tokenizer is required for prediction")
    
    # Tokenize and pad the text
    sequence = tokenizer.texts_to_sequences([text])
    padded_sequence = pad_sequences(sequence, maxlen=max_length)
    
    # Predict
    prediction = model.predict(padded_sequence)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative' if prediction[0][0] < 0.5 else 'neutral'
    return sentiment