import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
import re

# Target Cities in Kerala
# Commonfloor uses hyphenated city names usually, but "kochi", "trivandrum" etc might differ.
# I'll use a mapping or list.
CITIES = [
    'kochi', 'trivandrum', 'kozhikode', 'thiruvananthapuram' , 'pathanamthitta' , 'thrissur', 
    'kannur', 'alappuzha', 'kollam', 'palakkad'
    # Note: verify if these exist on commonfloor. 
    # if 404, we skip.
]

# Headers to mimic a browser
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept-Language': 'en-US,en;q=0.9',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

def get_url(city, page=1):
    # Commonfloor structure: https://www.commonfloor.com/{city}-property/for-rent?page={page}
    # Handling city name variations if needed.
    # Trivandrum -> thiruvananthapuram might be needed?
    # Let's try "trivandrum" first.
    return f"https://www.commonfloor.com/{city}-property/for-rent?page={page}"

def parse_price(price_text):
    # e.g. "30,000", "1.5 Lac"
    if not price_text: return 0
    clean_text = price_text.replace(',', '').strip()
    
    if 'Lac' in price_text or 'L' in price_text:
        try:
            val = float(re.findall(r"[\d\.]+", clean_text)[0])
            return int(val * 100000)
        except: return 0
    
    try:
        # Extract first number
        val = float(re.findall(r"[\d\.]+", clean_text)[0])
        return int(val)
    except:
        return 0

def fetch_listings(city, max_pages=3):
    listings = []
    print(f"Scraping {city}...")
    
    for page in range(1, max_pages + 1):
        url = get_url(city, page)
        print(f"  Fetching page {page}: {url}")
        
        try:
            response = requests.get(url, headers=HEADERS)
            if response.status_code != 200:
                print(f"  Failed to retrieve page {page} for {city}. Status: {response.status_code}")
                # Sometimes city name mismatch (e.g. trivandrum vs thiruvananthapuram)
                if page == 1 and response.status_code == 404:
                    print(f"  City {city} might use a different name or has no listings.")
                break
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Commonfloor cards
            cards = soup.find_all('div', class_='snb-tile')
            
            if not cards:
                print(f"  No listings found on page {page}. Stopping.")
                break
                
            print(f"  Found {len(cards)} listings on page {page}")

            for card in cards:
                try:
                    # Skip if it's purely an ad without property info (check title existence)
                    title_div = card.select_one('div.st_title')
                    if not title_div: continue
                    
                    # Title
                    title_a = title_div.select_one('h2 > a')
                    title = title_a.text.strip() if title_a else ""
                    
                    # Price
                    price_span = card.select_one('div.p_section > span.s_p')
                    price_text = price_span.text.strip() if price_span else "0"
                    rent = parse_price(price_text)
                    
                    if rent == 0: continue

                    # Location
                    loc_a = title_div.select_one('div.snbSubH > a')
                    location = loc_a.text.strip() if loc_a else ""
                    if not location:
                        # Fallback to title
                        location = title
                    
                    # Area
                    # Look for "Carpet Area" or "Super Built-up Area" in infodata
                    area = 0
                    info_divs = card.select('div.infodata')
                    for div in info_divs:
                        small = div.find('small')
                        if small and ('Area' in small.text):
                            span = div.find('span')
                            if span:
                                area_text = span.text.strip()
                                # e.g. "1040 sq.ft (96.62 sq.m)"
                                try:
                                    # Extract first number
                                    val = re.findall(r"[\d\.]+", area_text)[0]
                                    area = float(val)
                                except: pass
                                break
                    
                    # Bathrooms
                    bath = 0
                    for div in info_divs:
                        small = div.find('small')
                        if small and 'Bathroom' in small.text:
                            span = div.find('span')
                            if span:
                                try:
                                    bath = int(span.text.strip())
                                except: pass
                                break
                                
                    # Amenities
                    # .inforow .i_l > li
                    # class 'na' means not available
                    amenities_list = card.select('ul.i_l > li')
                    # We can store raw amenities or binary flags
                    parking = 0
                    power_backup = 0
                    pool = 0
                    security = 0
                    
                    for li in amenities_list:
                        text = li.text.strip()
                        is_available = 'na' not in li.get('class', [])
                        
                        if is_available:
                            if 'Parking' in text: parking = 1
                            if 'Power' in text: power_backup = 1
                            if 'Pool' in text: pool = 1
                            if 'Security' in text: security = 1
                            
                    # Furnishing
                    furnished = 'Unfurnished'
                    if 'Semi' in title or 'Semi' in card.text: furnished = 'Semi'
                    elif 'Fully' in title or 'Fully' in card.text: furnished = 'Fully'
                    
                    # Type (BHK/House/Villa)
                    # Infer from title
                    room_type_inferred = 'Single'
                    bhk = 0
                    if '1 BHK' in title: bhk = 1
                    elif '2 BHK' in title: bhk = 2
                    elif '3 BHK' in title: bhk = 3
                    elif '4' in title and 'BHK' in title: bhk = 4
                    
                    if bhk >= 3: room_type_inferred = 'Master'
                    if 'Villa' in title or 'House' in title: room_type_inferred = 'Master' # Broad categorization
                    
                    listings.append({
                        'city': city,
                        'location_raw': location,
                        'rent': rent,
                        'room_size': area,
                        'room_type_inferred': room_type_inferred,
                        'bhk': bhk,
                        'furnished': furnished,
                        'attached_bath': 1 if bath > 0 else 0, # Assume if bath listed it's attached/available
                        'bath_count': bath,
                        'parking': parking,
                        'power_backup': power_backup,
                        'pool': pool,
                        'security': security,
                        'source': 'commonfloor'
                    })
                    
                except Exception as e:
                    # print(f"Error parsing card: {e}")
                    continue
            
            # Respectful delay
            time.sleep(random.uniform(2, 4))
            
        except Exception as e:
            print(f"  Error fetching page {page}: {e}")
            break
            
    return listings

def scrape_all():
    all_data = []
    
    # Handle Trivandrum mapping if needed
    # Scrape loop
    real_cities = []
    for c in CITIES:
        if c == 'trivandrum': real_cities.append('trivandrum') # Try standard
        else: real_cities.append(c)
        
    for city in real_cities:
        data = fetch_listings(city, max_pages=3)
        if not data and city == 'trivandrum':
            # Retry with thiruvananthapuram
            print("Retrying with thiruvananthapuram...")
            data = fetch_listings('thiruvananthapuram', max_pages=3)
            
        all_data.extend(data)
        time.sleep(2) 
        
    # Save
    df = pd.DataFrame(all_data)
    print(f"Total listings scraped: {len(df)}")
    
    if not df.empty:
        df.to_csv('scraped_rent_data_raw.csv', index=False)
        print("Saved to scraped_rent_data_raw.csv")
        
        # Simple cleanup to match model
        # We need: location, room_type, room_size, ac, attached_bath, parking, kitchen, power_backup, etc.
        # We'll map what we have.
        # scraped data has: parking, power_backup, attached_bath (inferred), room_size, rent, furnished. 
        # Missing: ac, kitchen (assume 1), new amenities (wifi etc - random or 0).
        
    else:
        print("No data scraped.")

if __name__ == "__main__":
    scrape_all()
