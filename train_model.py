import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import random

# 1. Generate Synthetic Data (focused on Indian Cities context)
# We simulate detailed rental data to bootstrap the model
def generate_synthetic_data(n_samples=2000):
    np.random.seed(42)
    
    data = []
    
    # Base price mapping for locations (approximate monthly rent for 1 room)
    location_base_price = {
        'Kakkanad': 8000,
        'Edappally': 9000,
        'Palarivattom': 9500,
        'Kaloor': 9200,
        'MG Road': 10000,
        'Vyttila': 9500,
        'Aluva': 6000,
        'Kalamassery': 7000,
        'Fort Kochi': 12000, # Tourist premium
<<<<<<< HEAD
        'Other': 7500,
        # Add some Kerala cities base
        'Trivandrum': 8500,
        'Kochi': 8500
=======
        'Other': 7500
>>>>>>> d792f481d7c289764075a21142d5882cb649b70f
    }
    
    locations = list(location_base_price.keys())
    
    for _ in range(n_samples):
        loc = random.choice(locations)
        base = location_base_price[loc]
        
        # Features
        room_type = random.choice(['Single', 'Shared', 'Master', 'Studio'])
        room_size = random.randint(80, 400) # sqft
        
        # Amenities (0 or 1)
        ac = random.choice([0, 1])
        attached_bath = random.choice([0, 1])
        parking = random.choice([0, 1])
        kitchen = random.choice([0, 1])
        power_backup = random.choice([0, 1])
        
        # New Detailed Amenities
        wifi = random.choice([0, 1])
        tv = random.choice([0, 1])
        fridge = random.choice([0, 1])
        wardrobe = random.choice([0, 1])
        study_table = random.choice([0, 1])
        balcony = random.choice([0, 1])
        
        furnished = random.choice(['Unfurnished', 'Semi', 'Fully'])
        
        # Calculate Logic-based Rent (to make the model learn realistic patterns)
        rent = base
        
        # Room Type Multiplier
        if room_type == 'Single': rent *= 1.0
        elif room_type == 'Master': rent *= 1.4
        elif room_type == 'Studio': rent *= 1.6
        elif room_type == 'Shared': rent *= 0.6 # Rent per person usually
        
        # Size multiplier (small effect)
        rent += (room_size - 100) * 15 
        
        # Amenities Modifiers
        if ac: rent += 1500
        if attached_bath: rent += 1000
        if parking: rent += 500
        if kitchen: rent += 800
        if power_backup: rent += 400
        
        # New Amenities Values
        if wifi: rent += 500
        if tv: rent += 800
        if fridge: rent += 1000
        if wardrobe: rent += 500
        if study_table: rent += 300
        if balcony: rent += 500
        
        # Furnishing
        if furnished == 'Semi': rent += 1000
        elif furnished == 'Fully': rent += 2500
        
        # Random Noise (+- 10%)
        noise = random.uniform(0.9, 1.1)
        rent *= noise
        
        data.append({
            'location': loc,
            'room_type': room_type,
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
            'furnished': furnished,
            'rent': int(rent)
        })
        
<<<<<<< HEAD

    return pd.DataFrame(data)

def load_real_data(csv_path):
    try:
        df = pd.read_csv(csv_path)
        print(f"Loaded {len(df)} real samples.")
    except FileNotFoundError:
        print("Real data CSV not found. Using only synthetic.")
        return pd.DataFrame()
        
    # Map raw columns to model columns
    # Scraped: city, location_raw, rent, room_size, room_type_inferred, bhk, furnished, attached_bath, bath_count, parking, power_backup, pool, security, source
    # Model target cols: location, room_type, room_size, ac, attached_bath, parking, kitchen, power_backup, wifi, tv, fridge, wardrobe, study_table, balcony, furnished, rent
    
    processed_data = []
    
    for _, row in df.iterrows():
        # Clean Location
        loc = row['location_raw']
        if len(str(loc)) > 20 or 'Furnished' in str(loc):
            # Fallback to city or try to extract last word? 
            # Usually city is reliable from scraped data city col
            loc = row['city'].capitalize() 
        
        # Room Type
        rtype = row['room_type_inferred']
        if rtype not in ['Single', 'Master', 'Shared', 'Studio']:
            rtype = 'Single' # fallback
            
        # Furnished
        furn = row['furnished']
        if furn not in ['Unfurnished', 'Semi', 'Fully']:
            furn = 'Unfurnished'
            
        processed_data.append({
            'location': loc,
            'room_type': rtype,
            'room_size': float(row['room_size']) if row['room_size'] > 0 else 100.0,
            'ac': 0, # Missing in scraped
            'attached_bath': int(row['attached_bath']),
            'parking': int(row['parking']),
            'kitchen': 1, # Assume kitchen present in apartments
            'power_backup': int(row['power_backup']),
            'wifi': 0,
            'tv': 0,
            'fridge': 0,
            'wardrobe': 0,
            'study_table': 0,
            'balcony': 0, # could infer from description but keep simple
            'furnished': furn,
            'rent': int(row['rent'])
        })
        
    return pd.DataFrame(processed_data)

# 2. Train Model
def train():
    print("Generating synthetic data...")
    df_synthetic = generate_synthetic_data(2000) # Reduce synthetic if we have real
    
    print("Loading real data...")
    df_real = load_real_data('scraped_rent_data_raw.csv')
    
    # Combine
    if not df_real.empty:
        # Boost real data weight? Or just append. 
        # Let's append.
        df = pd.concat([df_synthetic, df_real], ignore_index=True)
        print(f"Total training samples: {len(df)} ({len(df_synthetic)} synthetic + {len(df_real)} real)")
    else:
        df = df_synthetic
        print(f"Total training samples: {len(df)}")
    
    # Preprocessing
    # Handle Location LabelEncoder with unseen labels in future?
    # We should handle unknown during inference, but for now fit on all.
    le_loc = LabelEncoder()
    # Normalize location case
    df['location'] = df['location'].astype(str).str.title().str.strip()
=======
    return pd.DataFrame(data)

# 2. Train Model
def train():
    print("Generating synthetic data...")
    df = generate_synthetic_data(5000)
    
    # Preprocessing
    le_loc = LabelEncoder()
>>>>>>> d792f481d7c289764075a21142d5882cb649b70f
    df['location_enc'] = le_loc.fit_transform(df['location'])
    
    le_type = LabelEncoder()
    df['room_type_enc'] = le_type.fit_transform(df['room_type'])
    
    le_furn = LabelEncoder()
    df['furnished_enc'] = le_furn.fit_transform(df['furnished'])
    
    # Features & Target
    X = df[['location_enc', 'room_type_enc', 'room_size', 'ac', 'attached_bath', 
            'parking', 'kitchen', 'power_backup', 'wifi', 'tv', 'fridge', 
            'wardrobe', 'study_table', 'balcony', 'furnished_enc']]
    y = df['rent']
    
    # Train
    print(f"Training RandomForest Regressor with {X.shape[1]} features...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Save Artifacts
    print("Saving model and encoders...")
    joblib.dump(model, 'rent_model.pkl')
    joblib.dump(le_loc, 'le_location.pkl')
    joblib.dump(le_type, 'le_room_type.pkl')
    joblib.dump(le_furn, 'le_furnished.pkl')
    
    print("✅ Model trained and saved successfully!")
    
    # Test a sample prediction
    # Kakkanad (assumed index known or we use transform), Single, 150sqft, AC, Bath, etc.
    # ideally we wrap prediction in a function but for script we just exit
    
if __name__ == '__main__':
    train()
