from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

import os

base_path = os.path.dirname(__file__)  # current file ka path
model_path = os.path.join(base_path, "model.lb")
columns_path = os.path.join(base_path, "model_columns.lb")

model = joblib.load(model_path)
model_columns = joblib.load(columns_path)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/project')
def project():
    return render_template('project.html')


# ✅ yahan 'predict' route banaya gaya hai (jo HTML se match karega)
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        try:
            # --- Get form inputs --- #
            brand = request.form['brand_name']
            owner = int(request.form['owner'])
            age = float(request.form['age'])
            power = float(request.form['power'])
            kms_driven = float(request.form['kms_driven'])

            # --- Input ko DataFrame me badla --- #
            input_data = pd.DataFrame({
                'brand': [brand],
                'owner': [owner],
                'age': [age],
                'power': [power],
                'kms_driven': [kms_driven]
            })

            # --- Same columns banana jaisa model ke training me tha --- #
            input_data = pd.get_dummies(input_data)
            input_data = input_data.reindex(columns=model_columns, fill_value=0)

            # --- Prediction --- #
            predicted_price = model.predict(input_data)[0]
            prediction = f"₹{predicted_price:,.0f}"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('project.html', prediction=prediction)

# ---------------- Run Flask ---------------- #
if __name__ == "__main__":
    app.run(debug=True)

