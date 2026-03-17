import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import joblib
import random
import os

# 1. Generate Synthetic Data with CORRECT pricing logic
def generate_synthetic_data(n_samples=15000):
    np.random.seed(42)
    data = []
    
    # Base price = price for a SINGLE room in that area
    location_base_price = {
        'Kakkanad': 8000, 'Edappally': 9000, 'Palarivattom': 9500, 'Kaloor': 9200,
        'MG Road': 10000, 'Vyttila': 9500, 'Aluva': 6000, 'Kalamassery': 7000,
        'Fort Kochi': 11000, 'Other': 6500,
        'Trivandrum': 8500, 'Kochi': 8500, 'Thrissur': 7500, 'Kozhikode': 7800,
        'Kannur': 6500, 'Alappuzha': 6000, 'Kollam': 6500, 'Palakkad': 5500,
        'Pathanamthitta': 5000, 'Ranni': 4500, 'Karukachal': 4800, 'Changanassery': 6000,
        'Kottayam': 7000, 'Pala': 5500,
        'Mumbai': 25000, 'Delhi': 15000, 'Bangalore': 16000, 'Hyderabad': 13000,
        'Chennai': 12000, 'Kolkata': 10000, 'Pune': 14000, 'Ahmedabad': 11000,
        'Surat': 9000, 'Jaipur': 8500
    }
    
    locations = list(location_base_price.keys())
    
    # Room types with their size multipliers (bigger room = more expensive)
    room_type_multipliers = {
        'Single': 1.0,    # small single room
        'Double': 1.6,    # bigger room with 2 beds / double bed
        'Triple': 2.1,    # 3 beds, large room
        'Quad': 2.8,      # 4 beds, very large room
        'Master': 1.8,    # premium master bedroom
        'Studio': 2.2     # self-contained studio apartment
    }
    
    # Bed type add-ons (No Bed is the baseline)
    bed_type_addon = {
        'No Bed': 0,
        'Single Bed': 800,
        'Double Bed': 2000,
        'Queen Bed': 3500,
        'King Bed': 5500,
        'Bunk Bed': 1200
    }
    
    room_types = list(room_type_multipliers.keys())
    bed_types = list(bed_type_addon.keys())
    
    for _ in range(n_samples):
        loc = random.choice(locations)
        base = location_base_price[loc]
        room_type = random.choice(room_types)
        bed_type = random.choice(bed_types)
        
        # Room size correlates with type
        if room_type == 'Single': room_size = random.randint(80, 150)
        elif room_type == 'Double': room_size = random.randint(120, 250)
        elif room_type == 'Master': room_size = random.randint(150, 300)
        elif room_type == 'Studio': room_size = random.randint(200, 400)
        elif room_type == 'Triple': room_size = random.randint(250, 450)
        elif room_type == 'Quad': room_size = random.randint(350, 600)
        else: room_size = random.randint(100, 300)
        
        # Amenities (binary)
        ac = random.choice([0, 1])
        attached_bath = random.choice([0, 1])
        parking = random.choice([0, 1])
        kitchen = random.choice([0, 1])
        power_backup = random.choice([0, 1])
        wifi = random.choice([0, 1])
        tv = random.choice([0, 1])
        fridge = random.choice([0, 1])
        wardrobe = random.choice([0, 1])
        study_table = random.choice([0, 1])
        balcony = random.choice([0, 1])
        furnished = random.choice(['Unfurnished', 'Semi', 'Fully'])
        
        # --- RENT CALCULATION ---
        rent = base * room_type_multipliers[room_type]
        
        # Bed type add-on
        rent += bed_type_addon[bed_type]
        
        # Size premium (larger rooms cost more per sqft)
        rent += (room_size - 100) * 12
        
        # Amenity add-ons (ALL amenities ADD to price, never subtract)
        if ac: rent += 1500
        if attached_bath: rent += 1000
        if parking: rent += 500
        if kitchen: rent += 800
        if power_backup: rent += 400
        if wifi: rent += 600
        if tv: rent += 500
        if fridge: rent += 700
        if wardrobe: rent += 500
        if study_table: rent += 400   # POSITIVE contribution
        if balcony: rent += 600
        
        # Furnishing
        if furnished == 'Semi': rent += 1500
        elif furnished == 'Fully': rent += 3500
        
        # Noise (+/- 8%)
        rent *= random.uniform(0.92, 1.08)
        
        # Cap rent to realistic range
        rent = max(2000, min(rent, 120000))
        
        data.append({
            'location': loc, 'room_type': room_type, 'bed_type': bed_type, 'room_size': room_size,
            'ac': ac, 'attached_bath': attached_bath, 'parking': parking,
            'kitchen': kitchen, 'power_backup': power_backup, 'wifi': wifi,
            'tv': tv, 'fridge': fridge, 'wardrobe': wardrobe,
            'study_table': study_table, 'balcony': balcony, 'furnished': furnished,
            'rent': int(rent)
        })
    return pd.DataFrame(data)

def load_real_data(csv_path):
    if not os.path.exists(csv_path): return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()
    processed = []
    valid_types = ['Single', 'Double', 'Triple', 'Quad', 'Master', 'Studio']
    for _, row in df.iterrows():
        try:
            rtype = row['room_type_inferred']
            if rtype not in valid_types:
                rtype = 'Double' if rtype == 'Shared' else 'Single'
            
            furn = row['furnished']
            if furn not in ['Unfurnished', 'Semi', 'Fully']:
                furn = 'Unfurnished'

            rent = int(row['rent'])
            if rent < 1000 or rent > 120000: continue  # skip bad data

            processed.append({
                'location': row['location_raw'], 'room_type': rtype, 'bed_type': 'No Bed',
                'room_size': float(row['room_size']) if row['room_size'] > 0 else 100.0,
                'ac': 0, 'attached_bath': int(row['attached_bath']), 'parking': int(row['parking']),
                'kitchen': 1, 'power_backup': int(row['power_backup']),
                'wifi': 0, 'tv': 0, 'fridge': 0, 'wardrobe': 0, 'study_table': 0, 'balcony': 0,
                'furnished': furn, 'rent': rent
            })
        except Exception:
            continue
    return pd.DataFrame(processed)

def load_external_data(csv_path):
    if not os.path.exists(csv_path): return pd.DataFrame()
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return pd.DataFrame()
    processed = []
    for _, row in df.iterrows():
        try:
            bhk = int(row['BHK'])
            # Stop mapping whole apartments to shared rooms! 
            # A 3BHK is not a 'Triple' room, it's a 3 bedroom house.
            if bhk > 2: 
                continue # Skip large apartments so they don't break room pricing logic
            
            if bhk == 2: rtype = 'Master'
            elif bhk == 0: rtype = 'Studio'
            else: rtype = 'Single'
            
            furn = str(row['Furnishing Status'])
            if 'Semi' in furn: fstatus = 'Semi'
            elif 'Unfurnished' in furn: fstatus = 'Unfurnished'
            else: fstatus = 'Fully'

            rent = int(row['Rent'])
            if rent < 1000 or rent > 120000: continue  # skip outliers

            size = float(row['Size'])
            # Don't let massive apartments pollute the 'No Bed' category and distort shared room pricing
            if size > 600: continue 

            processed.append({
                'location': str(row['City']).title(), 'room_type': rtype, 'bed_type': 'No Bed',
                'room_size': size,
                'ac': 0,
                'attached_bath': 0,
                'parking': 0, 'kitchen': 1, 'power_backup': 0,
                'wifi': 0,
                'tv': 0,
                'fridge': 0,
                'wardrobe': 0,
                'study_table': 0,
                'balcony': 0,
                'furnished': fstatus, 'rent': rent
            })
        except Exception:
            continue
    return pd.DataFrame(processed)

def train():
    print("Generating synthetic data...")
    df_syn = generate_synthetic_data(15000)
    print("Loading scraped & external data...")
    df_real = load_real_data('scraped_rent_data_raw.csv')
    df_ext = load_external_data('external_india_rent_data.csv')
    
    dfs = [df_syn]
    if not df_real.empty: dfs.append(df_real)
    if not df_ext.empty: dfs.append(df_ext)
    
    df = pd.concat(dfs, ignore_index=True)
    df['location'] = df['location'].astype(str).str.title().str.strip()
    
    # Final data cleaning
    df['rent'] = pd.to_numeric(df['rent'], errors='coerce')
    df = df.dropna()
    df = df[(df['rent'] >= 2000) & (df['rent'] <= 120000)].copy()
    
    print(f"Total clean samples: {len(df)}")
    df.to_csv('full_training_dataset.csv', index=False)
    
    # Distribution check
    print("\nRoom type average rents:")
    print(df.groupby('room_type')['rent'].mean().sort_values().to_string())
    print("\nBed type average rents:")
    print(df.groupby('bed_type')['rent'].mean().sort_values().to_string())
    
    # Encode categoricals
    le_loc = LabelEncoder()
    le_type = LabelEncoder()
    le_furn = LabelEncoder()
    le_bed = LabelEncoder()
    
    df['location_enc'] = le_loc.fit_transform(df['location'])
    df['room_type_enc'] = le_type.fit_transform(df['room_type'])
    df['furnished_enc'] = le_furn.fit_transform(df['furnished'])
    df['bed_type_enc'] = le_bed.fit_transform(df['bed_type'])
    
    feature_cols = ['location_enc', 'room_type_enc', 'bed_type_enc', 'room_size',
                    'ac', 'attached_bath', 'parking', 'kitchen', 'power_backup',
                    'wifi', 'tv', 'fridge', 'wardrobe', 'study_table', 'balcony', 'furnished_enc']
    
    X = df[feature_cols]
    y = df['rent']
    
    print(f"\nTraining RandomForest with {len(X)} samples...")
    model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
    model.fit(X, y)
    
    joblib.dump(model, 'rent_model.pkl')
    joblib.dump(le_loc, 'le_location.pkl')
    joblib.dump(le_type, 'le_room_type.pkl')
    joblib.dump(le_furn, 'le_furnished.pkl')
    joblib.dump(le_bed, 'le_bed_type.pkl')
    
    print("\n✅ Model trained and saved successfully!")
    
    # Quick sanity check
    print("\n--- Quick Sanity Check ---")
    le_kochi = le_loc.transform(['Kochi'])[0]
    for rtype in ['Single', 'Double', 'Triple', 'Quad', 'Master', 'Studio']:
        rtype_enc = le_type.transform([rtype])[0]
        bed_enc = le_bed.transform(['Single Bed'])[0]
        furn_enc = le_furn.transform(['Unfurnished'])[0]
        sample = pd.DataFrame([[le_kochi, rtype_enc, bed_enc, 150, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, furn_enc]], columns=feature_cols)
        pred = model.predict(sample)[0]
        print(f"  Kochi {rtype} (150sqft, Unfurnished): ₹{int(pred):,}")

if __name__ == '__main__':
    train()
