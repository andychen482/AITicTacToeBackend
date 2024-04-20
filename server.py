from python_speech_features import mfcc, delta
from scipy.io import wavfile
import numpy as np

from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load

# Load the model
model = load('model.joblib')
model2 = load('model2.joblib')


app = Flask(__name__)
CORS(app)

@app.route('/api', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    prediction = model.predict([data['gameState']])
    # print(prediction)
    return jsonify({'prediction': round(prediction[0])})

def extract_features_and_labels(audio_path):
    # Load the audio file
    samplerate, signal = wavfile.read(audio_path)
    
    # Window length and step in seconds
    winlen = 0.025
    winstep = 0.01
    
    # Calculate the frame length in samples
    frame_length_samples = int(winlen * samplerate)
    
    # Set nfft to the next power of two greater than frame length
    nfft = 2 ** np.ceil(np.log2(frame_length_samples)).astype(int)
    
    # Compute MFCC features from the audio signal
    mfcc_features = mfcc(signal, samplerate=samplerate, winlen=winlen, winstep=winstep, numcep=13, nfft=nfft, appendEnergy=True)
    print("MFCC Features Shape:", mfcc_features.shape)
    
    # Compute the first and second derivatives of the MFCC features
    delta_mfcc = delta(mfcc_features, 2)
    print("Delta MFCC Features Shape:", delta_mfcc.shape)
    delta_delta_mfcc = delta(delta_mfcc, 2)
    print("Delta Delta MFCC Features Shape:", delta_delta_mfcc.shape)
    
    # Concatenate the features together
    all_features = np.hstack((mfcc_features, delta_mfcc, delta_delta_mfcc))
    print("All Features Shape:", all_features.shape)
    
    # Calculate mean and standard deviation across all frames for each feature
    mean_features = np.mean(all_features, axis=0)
    std_features = np.std(all_features, axis=0)
    
    # Labels for the features
    feature_labels = []
    types = ['mean', 'std']
    feature_types = ['Log_energy', 'MFCC_0th', 'MFCC_1st', 'MFCC_2nd', 'MFCC_3rd', 'MFCC_4th', 'MFCC_5th', 'MFCC_6th', 'MFCC_7th', 'MFCC_8th', 'MFCC_9th', 'MFCC_10th', 'MFCC_11th', 'MFCC_12th']
    derivatives = ['', 'delta', 'delta_delta']
    
    # Generating labels for each type of feature
    for typ in types:
        for derivative in derivatives:
            for ft in feature_types:
                label = f"{typ}_{derivative}_{ft}".strip('_')
                feature_labels.append(label)
    
    # Combine the mean and standard deviation features
    combined_features = np.hstack((mean_features, std_features))
    
    return combined_features, feature_labels

@app.route('/pdprediction', methods=['POST'])
def pdpredict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file:
        file.save('temp.wav')
        features, labels = extract_features_and_labels('temp.wav')
        prediction = model2.predict([features])
        print("Prediction:", prediction[0])
        return jsonify({'prediction': int(prediction[0])})
    else:
        return jsonify({'error': 'No file'})

# for local testing
# if __name__ == '__main__':
#     app.run(port=3001, debug=True)