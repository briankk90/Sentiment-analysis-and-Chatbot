from flask import Flask, request, jsonify, render_template
from sentiment.model import predict_sentiment
from chatbot.agent import get_response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/sentiment', methods=['POST'])
def sentiment():
    text = request.json.get('text')
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    sentiment = predict_sentiment(text)
    return jsonify({'sentiment': sentiment})

@app.route('/api/chat', methods=['POST'])
def chat():
    user_input = request.json.get('text')
    if not user_input:
        return jsonify({'error': 'No input provided'}), 400
    response = get_response(user_input)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)