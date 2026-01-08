import requests
import json

url = "http://localhost:5001/predict_rent"
payload = {
    "location": "Kakkanad",
    "room_type": "Single",
    "room_size": 150,
    "amenities": {
        "wifi": True,
        "ac": True,
        "tv": True
    },
    "propertyAmenities": {
        "parking4w": True
    }
}
# Note: app.py expects flattened keys like 'wifi', 'ac' in the root or explicit handling.
# My app.py uses data.get('wifi') directly from the root of the JSON body?
# Let's check app.py again.
# Yes: wifi = int(data.get('wifi', 0))
# So the payload I send from Node.js is flattened:
# const payload = { wifi: ..., ac: ... }
# So I should test with THAT payload.

flat_payload = {
    "location": "Kakkanad",
    "room_type": "Single",
    "room_size": 150,
    "wifi": 1,
    "ac": 1,
    "tv": 1,
    "parking": 1,
    "furnished": "Semi"
}

try:
    response = requests.post(url, json=flat_payload)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except Exception as e:
    print("Error:", e)
