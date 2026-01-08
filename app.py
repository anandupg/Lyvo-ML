from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Load Models (Lazy loading or load on startup)
MODEL_PATH = 'rent_model.pkl'
LE_LOC_PATH = 'le_location.pkl'
LE_TYPE_PATH = 'le_room_type.pkl'
LE_FURN_PATH = 'le_furnished.pkl'

# Global variables
model = None
le_loc = None
le_type = None
le_furn = None

def load_models():
    global model, le_loc, le_type, le_furn
    if os.path.exists(MODEL_PATH):
        model = joblib.load(MODEL_PATH)
        le_loc = joblib.load(LE_LOC_PATH)
        le_type = joblib.load(LE_TYPE_PATH)
        le_furn = joblib.load(LE_FURN_PATH)
        print("✅ Models loaded successfully")
    else:
        print("❌ Model files not found. Please run train_model.py first.")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "price-predictor-ml"})

@app.route('/predict_rent', methods=['POST'])
def predict_rent():
    if not model:
        load_models()
        if not model:
            return jsonify({"error": "Model not trained"}), 500

    try:
        data = request.json
        
        # Extract features
        location = data.get('location', 'Other')
        room_type = data.get('room_type', 'Single')
        room_size = float(data.get('room_size', 100))
        ac = int(data.get('ac', 0))
        attached_bath = int(data.get('attached_bath', 0))
        parking = int(data.get('parking', 0))
        kitchen = int(data.get('kitchen', 0))
        power_backup = int(data.get('power_backup', 0))
        
        # New amenities
        wifi = int(data.get('wifi', 0))
        tv = int(data.get('tv', 0))
        fridge = int(data.get('fridge', 0))
        wardrobe = int(data.get('wardrobe', 0))
        study_table = int(data.get('study_table', 0))
        balcony = int(data.get('balcony', 0))
        
        furnished = data.get('furnished', 'Unfurnished')
        
        # Helpers to safely encode
        def safe_transform(encoder, value, default_val='Other'):
            try:
                return encoder.transform([value])[0]
            except ValueError:
                # Handle unknown labels (fallback to 'Other' or mode)
                # For location, fallback to 'Other' if it exists in encoder, else 0
                if 'Other' in encoder.classes_:
                    return encoder.transform(['Other'])[0]
                return 0 # Fallback index

        # Encode categorical
        loc_enc = safe_transform(le_loc, location)
        type_enc = safe_transform(le_type, room_type)
        furn_enc = safe_transform(le_furn, furnished)
        
        # Create Feature Vector
        # Order must match training: location, room_type, size, ac, bath, parking, kitchen, power, wifi, tv, fridge, wardrobe, study, balcony, furnished
        # Create Feature DataFrame to match training data structure and avoid warnings
        features_df = pd.DataFrame([{
            'location_enc': loc_enc,
            'room_type_enc': type_enc,
            'room_size': room_size,
            'ac': ac,
            'attached_bath': attached_bath,
            'parking': parking,
            'kitchen': kitchen,
            'power_backup': power_backup,
            'wifi': wifi,
            'tv': tv,
            'fridge': fridge,
            'wardrobe': wardrobe,
            'study_table': study_table,
            'balcony': balcony,
            'furnished_enc': furn_enc
        }])
        
        # Predict
        predicted_rent = model.predict(features_df)[0]
        
        # Range (+- 5-10%)
        min_rent = int(predicted_rent * 0.92)
        max_rent = int(predicted_rent * 1.08)
        
        return jsonify({
            "success": True,
            "predicted_rent": int(predicted_rent),
            "currency": "INR",
            "range": {
                "min": min_rent,
                "max": max_rent
            },
            "confidence": "High" # Placeholder
        })

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"success": False, "error": str(e)}), 400

if __name__ == '__main__':
    load_models()
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
