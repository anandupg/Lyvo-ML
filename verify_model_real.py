import joblib
import pandas as pd
import numpy as np

def verify():
    print("Loading model and encoders...")
    try:
        model = joblib.load('rent_model.pkl')
        le_loc = joblib.load('le_location.pkl')
        le_type = joblib.load('le_room_type.pkl')
        le_furn = joblib.load('le_furnished.pkl')
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return

    # Test Cases
    test_cases = [
        {
            'location': 'Kakkanad', 
            'room_type': 'Master', 
            'room_size': 1200, 
            'ac': 1, 
            'attached_bath': 1, 
            'parking': 1, 
            'kitchen': 1, 
            'power_backup': 1, 
            'wifi': 0, 'tv': 0, 'fridge': 0, 'wardrobe': 0, 'study_table': 0, 'balcony': 1,
            'furnished': 'Semi'
        },
        {
            'location': 'Trivandrum', 
            'room_type': 'Single', 
            'room_size': 500, 
            'ac': 0, 
            'attached_bath': 1, 
            'parking': 0, 
            'kitchen': 1, 
            'power_backup': 0, 
            'wifi': 0, 'tv': 0, 'fridge': 0, 'wardrobe': 0, 'study_table': 0, 'balcony': 0,
            'furnished': 'Unfurnished'
        }
    ]
    
    print("\nRunning Predictions:")
    for case in test_cases:
        # Transform
        try:
            loc_enc = le_loc.transform([case['location']])[0]
        except:
            print(f"Warning: Location {case['location']} not in training data. Using known fallback.")
            loc_enc = le_loc.transform(['Other'])[0]
            
        type_enc = le_type.transform([case['room_type']])[0]
        furn_enc = le_furn.transform([case['furnished']])[0]
        
        # Feature Vector
        features = [
            loc_enc, type_enc, case['room_size'], 
            case['ac'], case['attached_bath'], case['parking'], case['kitchen'], 
            case['power_backup'], case['wifi'], case['tv'], case['fridge'], 
            case['wardrobe'], case['study_table'], case['balcony'], 
            furn_enc
        ]
        
        pred = model.predict([features])[0]
        print(f"Case: {case['location']}, {case['room_type']}, {case['room_size']} sqft, {case['furnished']}")
        print(f"Predicted Rent: ₹{int(pred)}")
        print("-" * 30)

if __name__ == "__main__":
    verify()
