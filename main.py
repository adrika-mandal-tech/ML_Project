import pandas as pd
import numpy as np
import pickle
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder
import logging
import os
from functools import lru_cache
from werkzeug.middleware.proxy_fix import ProxyFix
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from warnings import filterwarnings

# Filter sklearn warnings
filterwarnings('ignore', category=UserWarning)

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Rate limiting
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Cache locations to avoid repeated CSV reading
@lru_cache(maxsize=1)
def get_locations():
    data = pd.read_csv("Cleaned_data.csv")
    return sorted(data['location'].unique())


# Load model (with error handling)
def load_model():
    model_files = ["RidgeModel.pk1", "RidgeModel.pkl", "ridge_model.pkl"]
    for model_file in model_files:
        if os.path.exists(model_file):
            try:
                with open(model_file, "rb") as f:
                    model = pickle.load(f)
                    logging.info(f"Successfully loaded model from {model_file}")
                    return model
            except Exception as e:
                logging.error(f"Error loading model from {model_file}: {str(e)}")
    raise FileNotFoundError("Could not find or load any model file")


# Load data and model at startup
try:
    data = pd.read_csv("Cleaned_data.csv")
    pipe = load_model()

    # Create and fit label encoder
    location_encoder = LabelEncoder()
    location_encoder.fit(data['location'])

    # Pre-warm the cache
    get_locations()

except Exception as e:
    logging.error(f"Initialization error: {str(e)}", exc_info=True)
    raise


@app.route('/')
def index():
    try:
        locations = get_locations()
        return render_template('index.html', locations=locations)
    except Exception as e:
        logging.error(f"Error in index route: {str(e)}", exc_info=True)
        return render_template('error.html'), 500


@app.route('/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    try:
        # Get and validate form data
        location = request.form.get("location", "").strip()
        bhk = request.form.get("bhk", "").strip()
        bath = request.form.get("bath", "").strip()
        sqft = request.form.get("total_sqft", "").strip()

        if not all([location, bhk, bath, sqft]):
            return jsonify({"error": "All fields are required"}), 400

        try:
            bhk = int(bhk)
            bath = int(bath)
            sqft = float(sqft)

            # Validate ranges
            if not (1 <= bhk <= 10):
                return jsonify({"error": "BHK must be between 1 and 10"}), 400
            if not (1 <= bath <= 10):
                return jsonify({"error": "Bath must be between 1 and 10"}), 400
            if not (300 <= sqft <= 10000):
                return jsonify({"error": "Square feet must be between 300 and 10,000"}), 400

        except ValueError as ve:
            return jsonify({"error": f"Invalid numerical value: {str(ve)}"}), 400

        # Encode location
        try:
            location_encoded = location_encoder.transform([location])[0]
        except ValueError:
            return jsonify({"error": "Invalid location specified"}), 400

        # Prepare input as DataFrame with correct feature names
        input_df = pd.DataFrame([[location_encoded, sqft, bath, bhk]],
                                columns=['location', 'total_sqft', 'bath', 'bhk'])

        # Make prediction
        prediction = pipe.predict(input_df)[0] * 1e5
        formatted_pred = f"₹{np.round(prediction, 2):,}"
        return formatted_pred

    except Exception as e:
        logging.error(f"Prediction error: {str(e)}", exc_info=True)
        return jsonify({"error": "An error occurred during prediction"}), 500


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001, debug=False)