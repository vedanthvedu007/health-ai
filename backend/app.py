from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load model + encoder + features
model = pickle.load(open("model.pkl", "rb"))
le = pickle.load(open("encoder.pkl", "rb"))
features = pickle.load(open("features.pkl", "rb"))

# ---------------- NLP (text → symptoms) ----------------
def extract_symptoms(text):
    text = text.lower()
    symptoms = []

    if "fever" in text:
        symptoms.append("fever")
    if "headache" in text or "head hurts" in text:
        symptoms.append("headache")
    if "chest pain" in text:
        symptoms.append("chest pain")
    if "breathing" in text or "breath" in text:
        symptoms.append("breathing difficulty")
    if "fatigue" in text or "tired" in text:
        symptoms.append("fatigue")

    return symptoms

# ---------------- Feature Engineering ----------------
def preprocess(symptoms_list):
    input_data = [0] * len(features)

    for sym in symptoms_list:
        if sym in features:
            idx = features.index(sym)
            input_data[idx] = 1

    return np.array([input_data])

# ---------------- Safety Rules ----------------
def apply_rules(symptoms):
    if "chest pain" in symptoms or "breathing difficulty" in symptoms:
        return "HIGH", True
    return None, False

# ---------------- Triage Recommendation ----------------
def get_recommendation(risk):
    if risk == "LOW":
        return "Home care is sufficient"
    elif risk == "MEDIUM":
        return "Visit a doctor or clinic"
    elif risk == "HIGH":
        return "Go to emergency immediately"

# ---------------- API ----------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # NLP input
    text = data.get("text", "")
    symptoms = extract_symptoms(text)

    # Rule-based override
    rule_risk, emergency = apply_rules(symptoms)
    if rule_risk:
        return jsonify({
            "symptoms": symptoms,
            "risk": rule_risk,
            "emergency": emergency,
            "recommendation": get_recommendation(rule_risk),
            "reason": "Critical symptoms detected (rule-based)"
        })

    # ML prediction
    processed = preprocess(symptoms)
    pred = model.predict(processed)
    risk = le.inverse_transform(pred)[0]

    return jsonify({
        "symptoms": symptoms,
        "risk": risk,
        "emergency": False,
        "recommendation": get_recommendation(risk),
        "reason": "Predicted using ML model"
    })

# Optional home route
@app.route("/")
def home():
    return "AI Triage System Running"

if __name__ == "__main__":
    app.run(debug=True)
    