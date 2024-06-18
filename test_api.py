import requests

url = 'http://127.0.0.1:8000/predict'
headers = {'Content-Type': 'application/json'}
data = {
    "online_order": "Yes",
    "book_table": "No",
    "votes": 775,
    "rest_type": "Casual Dining",
    "cost": 800,
    "type": "Buffet",
    "city": "Banashankari"
}

response = requests.post(url, headers=headers, json=data)

print("Response status code:", response.status_code)

# Attempt to parse the response as JSON
try:
    response_json = response.json()
    print("Response JSON:", response_json)
except ValueError:
    print("Response text:", response.text)
