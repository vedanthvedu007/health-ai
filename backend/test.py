import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "text": "I have fever and headache"
}

res = requests.post(url, json=data)
print(res.json())