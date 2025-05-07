import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Embedding, LSTM, Dense
import os

def build_model(vocab_size, embedding_dim=100, lstm_units=128, max_len=100):
    """
    Build and compile an LSTM-based neural network for sentiment analysis.
    
    Args:
        vocab_size (int): Size of the vocabulary for the embedding layer
        embedding_dim (int): Dimension of the embedding vectors
        lstm_units (int): Number of LSTM units
        max_len (int): Maximum length of input sequences
    
    Returns:
        tf.keras.Model: Compiled LSTM model
    """
    model = Sequential([
        Embedding(vocab_size, embedding_dim, input_length=max_len),
        LSTM(lstm_units, return_sequences=False),
        Dense(64, activation='relu'),
        Dense(3, activation='softmax')  # 3 classes: positive, negative, neutral
    ])
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])
    return model

def load_trained_model(model_path='data/trained_model/sentiment_model.h5'):
    """
    Load a pre-trained sentiment model.
    
    Args:
        model_path (str): Path to the saved model
    
    Returns:
        tf.keras.Model: Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    return load_model(model_path)

def predict_sentiment(text, tokenizer=None, model_path='data/trained_model/sentiment_model.h5', max_len=100):
    """
    Predict sentiment for a given text input.
    
    Args:
        text (str): Input text to classify
        tokenizer: Tokenizer fitted on training data
        model_path (str): Path to the trained model
        max_len (int): Maximum length of input sequences
    
    Returns:
        str: Predicted sentiment ('positive', 'negative', 'neutral')
    """
    model = load_trained_model(model_path)
    
    # Preprocess the input text
    if tokenizer is None:
        raise ValueError("Tokenizer is required for prediction")
    
    sequence = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(sequence, maxlen=max_len)
    
    # Predict
    prediction = model.predict(padded, verbose=0)
    labels = ['negative', 'neutral', 'positive']
    predicted_label = labels[tf.argmax(prediction, axis=1).numpy()[0]]
    
    return predicted_label