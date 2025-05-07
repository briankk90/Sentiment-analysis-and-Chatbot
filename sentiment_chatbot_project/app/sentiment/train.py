import pandas as pd
import os
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from .model import build_model
from .preprocess import preprocess_texts, encode_labels

def train_model(data_path='data/reviews.csv', model_save_path='data/trained_model/sentiment_model.h5'):
    """
    Train the sentiment analysis model and save it.
    
    Args:
        data_path (str): Path to the CSV dataset
        model_save_path (str): Path to save the trained model
    """
    # Load dataset
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    data = pd.read_csv(data_path)
    texts, labels = data['review'], data['sentiment']
    
    # Preprocess texts and labels
    X, tokenizer = preprocess_texts(texts)
    y = encode_labels(labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Build model
    vocab_size = len(tokenizer.word_index) + 1
    model = build_model(vocab_size=vocab_size)
    
    # Define callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    # Train model
    model.fit(X_train, y_train,
              validation_data=(X_test, y_test),
              epochs=10,
              batch_size=32,
              callbacks=[early_stopping],
              verbose=1)
    
    # Save model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")

if __name__ == '__main__':
    train_model()