import requests
import json

try:
    response = requests.post(
        'http://localhost:5001/predict_rent',
        json={
            "location": "Kakkanad",
            "room_type": "Single",
            "room_size": 150,
            "ac": 1,
            "attached_bath": 1
        }
    )
    print("Status Code:", response.status_code)
    print("Response:", json.dumps(response.json(), indent=2))
except Exception as e:
    print("Error:", e)
