from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load models
diabetes_model = joblib.load("diabetes_model.pkl")
heart_model = joblib.load("heart_model.pkl")
liver_model = joblib.load("liver_model.pkl")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/diabetes', methods=['GET', 'POST'])
def diabetes():
    if request.method == 'POST':
        try:
            data = [float(x) for x in request.form.values()]
            prediction = diabetes_model.predict([data])[0]
            output = "Diabetes" if prediction == 1 else "Not Diabetes"
            return render_template('diabetes.html', prediction_text=output)
        except Exception as e:
            return render_template('diabetes.html', prediction_text=f"Error: {str(e)}")
    return render_template('diabetes.html')


@app.route('/heart', methods=['GET', 'POST'])
def heart():
    if request.method == 'POST':
        try:
            data = [float(x) for x in request.form.values()]
            prediction = heart_model.predict([data])[0]
            output = "Heart Disease Detected" if prediction == 1 else "No Heart Disease"
            return render_template('heart.html', prediction_text=output)
        except Exception as e:
            return render_template('heart.html', prediction_text=f"Error: {str(e)}")
    return render_template('heart.html')


@app.route('/liver', methods=['GET', 'POST'])
def liver():
    if request.method == 'POST':
        try:
            data = [float(x) for x in request.form.values()]
            prediction = liver_model.predict([data])[0]
            output = "Liver Disease Detected" if prediction == 1 else "No Liver Disease"
            return render_template('liver.html', prediction_text=output)
        except Exception as e:
            return render_template('liver.html', prediction_text=f"Error: {str(e)}")
    return render_template('liver.html')


if __name__ == '__main__':
    app.run(debug=True)

