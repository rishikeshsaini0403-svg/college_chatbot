from flask import Flask, render_template, request, jsonify
import pickle, json, numpy as np, os

app = Flask(__name__)

# Load dataset
data = json.load(open("data/intents.json"))

# Load model if exists, otherwise fallback
model_path = "model/intent_model.pkl"
vec_path = "model/vectorizer.pkl"

model = None
vectorizer = None

try:
    if os.path.getsize(model_path) > 0:
        model = pickle.load(open(model_path, "rb"))
    if os.path.getsize(vec_path) > 0:
        vectorizer = pickle.load(open(vec_path, "rb"))
except:
    model = None
    vectorizer = None

def chatbot_response(text):
    text_lower = text.lower()

    # If trained model exists
    if model and vectorizer:
        X = vectorizer.transform([text])
        intent = model.predict(X)[0]
        for item in data["intents"]:
            if item["intent"] == intent:
                return np.random.choice(item["responses"])

    # Fallback keyword matching
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            if pattern.lower() in text_lower:
                return np.random.choice(intent["responses"])

    return "Sorry, I didn’t understand that. Please try again."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg")
    reply = chatbot_response(msg)
    return jsonify({"reply": reply})

if __name__ == "__main__":
    app.run(debug=True)
