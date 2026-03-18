from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

# Use absolute paths so the server works regardless of cwd
BASE_DIR = os.path.dirname(__file__)

# Global variables - Load at startup
MODEL_PATH = os.path.join(BASE_DIR, 'rent_model.pkl')
LE_LOC_PATH = os.path.join(BASE_DIR, 'le_location.pkl')
LE_TYPE_PATH = os.path.join(BASE_DIR, 'le_room_type.pkl')
LE_FURN_PATH = os.path.join(BASE_DIR, 'le_furnished.pkl')
LE_BED_PATH = os.path.join(BASE_DIR, 'le_bed_type.pkl')
PRIORITY_MODEL_PATH = os.path.join(BASE_DIR, 'priority_model.pkl')

def load_models():
    if os.path.exists(MODEL_PATH):
        try:
            m = joblib.load(MODEL_PATH)
            loc = joblib.load(LE_LOC_PATH)
            typ = joblib.load(LE_TYPE_PATH)
            furn = joblib.load(LE_FURN_PATH)
            bed = joblib.load(LE_BED_PATH)
            print("✅ Models loaded successfully")
            return m, loc, typ, furn, bed
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            return None, None, None, None, None
    else:
        print("❌ Model files not found. Please run train_model.py first.")
        return None, None, None, None, None

def load_priority_model():
    if os.path.exists(PRIORITY_MODEL_PATH):
        try:
            m = joblib.load(PRIORITY_MODEL_PATH)
            print("✅ Priority model loaded successfully")
            return m
        except Exception as e:
            print(f"❌ Error loading priority model: {e}")
            return None
    else:
        print("⚠️  priority_model.pkl not found. Run train_priority_model.py first.")
        return None

model, le_loc, le_type, le_furn, le_bed = load_models()
priority_model = load_priority_model()

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy", "service": "price-predictor-ml"})

@app.route('/predict_rent', methods=['POST'])
def predict_rent():
    if not model:
        return jsonify({"success": False, "error": "Model not loaded on server. Please check server logs."}), 500

    try:
        data = request.json or {}

        # Extract features
        location = data.get('location', 'Other')
        room_type = data.get('room_type', 'Single')
        bed_type = data.get('bed_type', 'Single Bed')
        room_size = float(data.get('room_size', 100))

        # Helpers for boolean/flag values
        def _bool_flag(v):
            if isinstance(v, bool):
                return int(v)
            if isinstance(v, str):
                if v.strip().lower() in ('1', 'true', 'yes', 'y', 'on'):
                    return 1
                return 0
            try:
                return int(v)
            except Exception:
                return 0
        
        # Enforce realistic minimum sizes so the RF model doesn't break
        min_sizes = {
            'Single': 80,
            'Double': 120,
            'Master': 150,
            'Studio': 200,
            'Triple': 250,
            'Quad': 350
        }
        min_required = min_sizes.get(room_type, 100)
        if room_size < min_required:
            room_size = min_required

        ac = _bool_flag(data.get('ac', 0))
        attached_bath = _bool_flag(data.get('attached_bath', 0))
        parking = _bool_flag(data.get('parking', 0))
        kitchen = _bool_flag(data.get('kitchen', 0))
        power_backup = _bool_flag(data.get('power_backup', 0))
        
        # New amenities
        wifi = _bool_flag(data.get('wifi', 0))
        tv = _bool_flag(data.get('tv', 0))
        fridge = _bool_flag(data.get('fridge', 0))
        wardrobe = _bool_flag(data.get('wardrobe', 0))
        study_table = _bool_flag(data.get('study_table', 0))
        balcony = _bool_flag(data.get('balcony', 0))
        
        furnished = data.get('furnished', 'Unfurnished')
        
        # Helpers to safely encode
        def safe_transform(encoder, value, default_val='Other'):
            try:
                return encoder.transform([value])[0]
            except Exception:
                # Handle unknown labels (fallback to default, then 'Other', then first class)
                if default_val in encoder.classes_:
                    return encoder.transform([default_val])[0]
                if 'Other' in encoder.classes_:
                    return encoder.transform(['Other'])[0]
                # fallback to first known class (should always exist for a fitted encoder)
                return encoder.transform([encoder.classes_[0]])[0]

        # Encode categorical
        loc_enc = safe_transform(le_loc, location)
        type_enc = safe_transform(le_type, room_type)
        furn_enc = safe_transform(le_furn, furnished)
        bed_enc = safe_transform(le_bed, bed_type, default_val='Single Bed')
        
        # Create Feature Vector as DataFrame with column names matching training data
        feature_names = [
            'location_enc', 'room_type_enc', 'bed_type_enc', 'room_size',
            'ac', 'attached_bath', 'parking', 'kitchen', 'power_backup',
            'wifi', 'tv', 'fridge', 'wardrobe', 'study_table', 'balcony',
            'furnished_enc'
        ]
        features = pd.DataFrame([[
            loc_enc,
            type_enc,
            bed_enc,
            room_size,
            ac,
            attached_bath,
            parking,
            kitchen,
            power_backup,
            wifi,
            tv,
            fridge,
            wardrobe,
            study_table,
            balcony,
            furn_enc
        ]], columns=feature_names)
        
        # Predict
        predicted_rent = model.predict(features)[0]
        
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

@app.route('/predict_priority', methods=['POST'])
def predict_priority():
    if not priority_model:
        # Graceful fallback — caller should handle this
        return jsonify({"success": False, "priority": "medium", "confidence": 0.0,
                        "error": "Priority model not loaded. Run train_priority_model.py first."}), 200

    try:
        data = request.json or {}
        title = str(data.get('title', '')).strip()
        description = str(data.get('description', '')).strip()
        category = str(data.get('category', '')).strip()

        if not title and not description:
            return jsonify({"success": False, "priority": "medium", "confidence": 0.0,
                            "error": "title or description required"}), 400

        # Combine inputs the same way the training script does
        combined = f"{title} {description} {category}".strip()

        priority = priority_model.predict([combined])[0]
        probas = priority_model.predict_proba([combined])[0]
        classes = priority_model.classes_.tolist()
        confidence = float(max(probas))

        return jsonify({
            "success": True,
            "priority": priority,
            "confidence": round(confidence, 3),
            "probabilities": dict(zip(classes, [round(float(p), 3) for p in probas]))
        })

    except Exception as e:
        print("Priority prediction error:", str(e))
        return jsonify({"success": False, "priority": "medium", "confidence": 0.0,
                        "error": str(e)}), 400


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)
