from flask import Flask, render_template, request
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression

app = Flask(__name__)
model = pickle.load(open('house_regmodel.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    input_features = [float(x) for x in request.form.values()]
    prediction = model.predict([input_features])[0]
    return render_template('index.html', prediction_text=f'Predicted Price: ${prediction * 100000:.2f}')

if __name__ == "__main__":
    app.run(debug=True)
