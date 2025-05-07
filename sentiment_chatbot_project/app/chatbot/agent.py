import json
from sentiment.model import predict_sentiment
from .nlp import process_input

# Load predefined responses
with open('app/chatbot/responses.json') as f:
    responses = json.load(f)

def match_intent(tokens, entities):
    """
    Match user input to an intent based on tokens and entities.
    
    Args:
        tokens (list): List of tokenized words
        entities (list): List of (entity_text, entity_label) tuples
    
    Returns:
        str: Matched intent key or 'default'
    """
    if any(token in ['hi', 'hello', 'hey'] for token in tokens):
        return 'greeting'
    if any(token in ['issue', 'problem', 'complaint'] for token in tokens):
        return 'complaint'
    return 'default'

def get_response(user_input):
    """
    Generate a chatbot response based on user input, incorporating sentiment analysis.
    
    Args:
        user_input (str): User input text
    
    Returns:
        str: Chatbot response
    """
    sentiment = predict_sentiment(user_input)
    tokens, entities = process_input(user_input)
    intent = match_intent(tokens, entities)
    response = responses.get(intent, responses['default'])
    
    # Adjust response based on sentiment
    if sentiment == 'negative':
        response = f"I'm sorry to hear that. {response}"
    elif sentiment == 'positive':
        response = f"That's great to hear! {response}"
    
    return response