import pickle
from flask import Flask, request, jsonify, render_template, url_for, app
import numpy as np
import pandas as pd

app = Flask(__name__) # __name__ is the starting point of the application

## load the model
model = pickle.load(open('regmodel.pkl', 'rb'))
scalar = pickle.load(open('scaler.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scalar.transform(np.array(list(data.values())).reshape(1, -1))
    prediction = model.predict(new_data)
    print(prediction[0])
    return jsonify(prediction[0])

if __name__ == "__main__":
    app.run(debug=True)