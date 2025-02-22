from flask import Flask, request, render_template
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load model
model = joblib.load('car_prediction.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    result = ''
    return render_template('index.html', result=result)  

@app.route('/predict', methods=['POST','GET'])
def predict():
    try:
        # Ambil input dari form
        year = int(request.form['year'])
        engine_size = float(request.form['engine_size'])  # Harus float
        mileage = int(request.form['mileage'])
        doors = int(request.form['doors'])

        # Bentuk input sebagai array 2D
        new_data = np.array([[year, engine_size, mileage, doors]])
        
        # Standarisasi data
        new_data_scaled = scaler.transform(new_data)

        # Prediksi harga
        predicted_price = model.predict(new_data_scaled)[0]

        # Format output seperti yang diinginkan
        result = f"${predicted_price:,.2f}"

    except Exception as e:
        result = f"Error: {e}"

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
