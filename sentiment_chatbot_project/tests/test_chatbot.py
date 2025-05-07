import unittest
from chatbot.agent import get_response, match_intent
from chatbot.nlp import process_input

class TestChatbot(unittest.TestCase):
    def setUp(self):
        """Set up sample inputs for testing."""
        self.greeting_input = "Hi there"
        self.complaint_input = "I have a problem with my order"
        self.default_input = "What's the weather like?"

    def test_process_input(self):
        """Test NLP processing for tokens and entities."""
        tokens, entities = process_input(self.greeting_input)
        self.assertTrue(isinstance(tokens, list))
        self.assertTrue(isinstance(entities, list))
        self.assertIn("hi", tokens)
        self.assertIn("there", tokens)

    def test_match_intent_greeting(self):
        """Test intent matching for greeting."""
        tokens, _ = process_input(self.greeting_input)
        intent = match_intent(tokens, [])
        self.assertEqual(intent, 'greeting')

    def test_match_intent_complaint(self):
        """Test intent matching for complaint."""
        tokens, _ = process_input(self.complaint_input)
        intent = match_intent(tokens, [])
        self.assertEqual(intent, 'complaint')

    def test_match_intent_default(self):
        """Test intent matching for default case."""
        tokens, _ = process_input(self.default_input)
        intent = match_intent(tokens, [])
        self.assertEqual(intent, 'default')

    def test_get_response_greeting(self):
        """Test response generation for greeting."""
        response = get_response(self.greeting_input)
        self.assertIn("Hello", response)

    def test_get_response_complaint(self):
        """Test response generation for complaint."""
        response = get_response(self.complaint_input)
        self.assertIn("sorry", response.lower())  # Expect empathetic response due to negative sentiment

if __name__ == '__main__':
    unittest.main()