from flask import Flask, render_template, request
import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

# Preprocess function
def preprocess_data(data):
    # Convert categorical variables to one-hot encoding
    data = pd.get_dummies(data)
    # Standardize features
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    return data_scaled

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(x) for x in request.form.values()]
    data = [features]
    data_scaled = preprocess_data(data)
    prediction = model.predict(data_scaled)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)