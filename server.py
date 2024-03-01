from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from joblib import load

# Load the model
model = load('model.joblib')


app = Flask(__name__)
CORS(app)

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['gameState']])
    return jsonify({'prediction': round(prediction[0])})

# for local testing
# if __name__ == '__main__':
#     app.run(port=3001, debug=True)