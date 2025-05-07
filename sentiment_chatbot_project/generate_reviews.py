import pandas as pd
import random
from faker import Faker

def generate_fake_reviews(n=1000):
    """
    Generate a fake dataset of customer reviews with sentiments.
    
    Args:
        n (int): Number of reviews to generate
    
    Returns:
        pd.DataFrame: DataFrame with 'review' and 'sentiment' columns
    """
    fake = Faker()
    sentiments = ['positive', 'negative', 'neutral']
    
    # Sample review templates
    positive_templates = [
        "I love this product! It's absolutely {adj}.",
        "Great experience with this service, very {adj}!",
        "Highly recommend this, it’s {adj} and works perfectly."
    ]
    negative_templates = [
        "Really disappointed with this, it’s {adj}.",
        "Terrible service, completely {adj}.",
        "Not worth the money, very {adj}."
    ]
    neutral_templates = [
        "The product is okay, fairly {adj}.",
        "It works as expected, nothing {adj} about it.",
        "Decent service, but could be more {adj}."
    ]
    
    adjectives = ['amazing', 'reliable', 'user-friendly', 'poor', 'frustrating', 'average', 'standard']
    
    data = {'review': [], 'sentiment': []}
    
    for _ in range(n):
        sentiment = random.choice(sentiments)
        if sentiment == 'positive':
            template = random.choice(positive_templates)
        elif sentiment == 'negative':
            template = random.choice(negative_templates)
        else:
            template = random.choice(neutral_templates)
        
        review = template.format(adj=random.choice(adjectives))
        data['review'].append(review)
        data['sentiment'].append(sentiment)
    
    return pd.DataFrame(data)

if __name__ == '__main__':
    # Generate 1000 fake reviews and save to data/reviews.csv
    df = generate_fake_reviews(1000)
    df.to_csv('data/reviews.csv', index=False)
    print("Generated data/reviews.csv with 1000 fake reviews.")