import unittest
from app.main import app
import json

class TestAPI(unittest.TestCase):
    def setUp(self):
        """Set up Flask test client."""
        self.app = app.test_client()
        self.app.testing = True

    def test_index_route(self):
        """Test the index route for correct status and content."""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'Sentiment Analysis & Chatbot', response.data)

    def test_sentiment_api_valid_input(self):
        """Test sentiment API with valid input."""
        response = self.app.post('/api/sentiment',
                               data=json.dumps({'text': 'Great product!'}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertEqual(data['sentiment'], 'positive')

    def test_sentiment_api_no_text(self):
        """Test sentiment API with missing text."""
        response = self.app.post('/api/sentiment',
                               data=json.dumps({}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'No text provided')

    def test_chat_api_valid_input(self):
        """Test chat API with valid input."""
        response = self.app.post('/api/chat',
                               data=json.dumps({'text': 'Hi there'}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIn('Hello', data['response'])

    def test_chat_api_no_text(self):
        """Test chat API with missing text."""
        response = self.app.post('/api/chat',
                               data=json.dumps({}),
                               content_type='application/json')
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.data)
        self.assertEqual(data['error'], 'No input provided')

if __name__ == '__main__':
    unittest.main()