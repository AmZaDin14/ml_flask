from flask import Flask, render_template, request
import joblib  # Untuk memuat model
import numpy as np

app = Flask(__name__)

# Load model
model = joblib.load("model.pkl")  # Sesuaikan dengan nama dan lokasi file model Anda


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Ambil nilai input dari form
        input_data = [float(x) for x in request.form.values()]
        input_array = np.array(input_data).reshape(1, -1)

        # Lakukan prediksi menggunakan model
        prediction = model.predict(input_array)

        return render_template(
            "index.html",
            features_text=f"{input_data}",
            prediction_text=f"Prediksi Harga Rumah: $ {prediction[0]:,.2f}",
        )


if __name__ == "__main__":
    app.run(debug=True)
