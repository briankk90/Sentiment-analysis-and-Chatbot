import spacy

# Load the English NLP model
nlp = spacy.load('en_core_web_sm')

def process_input(text):
    """
    Process user input using spaCy for tokenization, POS tagging, and entity recognition.
    Returns tokens and entities for intent matching.
    
    Args:
        text (str): User input text
    
    Returns:
        tuple: (list of tokens, list of (entity_text, entity_label) tuples)
    """
    doc = nlp(text.lower())
    tokens = [token.text for token in doc]
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return tokens, entities