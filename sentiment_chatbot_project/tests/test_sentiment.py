import unittest
import numpy as np
from sentiment.preprocess import preprocess_texts, encode_labels
from sentiment.model import predict_sentiment, build_model
from tensorflow.keras.preprocessing.text import Tokenizer

class TestSentiment(unittest.TestCase):
    def setUp(self):
        """Set up a small dataset and tokenizer for testing."""
        self.texts = ["This is great!", "Terrible experience.", "It's okay."]
        self.labels = ["positive", "negative", "neutral"]
        self.max_words = 100
        self.max_len = 10
        self.padded, self.tokenizer = preprocess_texts(self.texts, max_words=self.max_words, max_len=self.max_len)
        self.encoded_labels = encode_labels(self.labels)

    def test_preprocess(self):
        """Test text preprocessing for correct shape and tokenizer output."""
        self.assertEqual(self.padded.shape, (len(self.texts), self.max_len))
        self.assertTrue(isinstance(self.tokenizer, Tokenizer))
        self.assertGreater(len(self.tokenizer.word_index), 0)

    def test_encode_labels(self):
        """Test label encoding for correct numerical values."""
        expected = np.array([2, 0, 1])  # positive: 2, negative: 0, neutral: 1
        np.testing.assert_array_equal(self.encoded_labels, expected)

    def test_build_model(self):
        """Test model building for correct output shape."""
        vocab_size = len(self.tokenizer.word_index) + 1
        model = build_model(vocab_size, max_len=self.max_len)
        input_data = np.random.randint(0, vocab_size, (1, self.max_len))
        output = model.predict(input_data, verbose=0)
        self.assertEqual(output.shape, (1, 3))  # 3 classes: positive, negative, neutral

if __name__ == '__main__':
    unittest.main()