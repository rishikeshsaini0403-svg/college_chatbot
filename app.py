from flask import Flask, render_template, request, jsonify
import pickle, json, numpy as np, os

app = Flask(__name__, template_folder="templates")

# Load dataset
data = json.load(open("Data/intents.json"))

# Load model if exists, otherwise fallback
model_path = "model/intent_model.pkl"
vec_path = "model/vectorizer.pkl"

model = None
vectorizer = None

try:
    if os.path.exists(model_path) and os.path.getsize(model_path) > 0:
        model = pickle.load(open(model_path, "rb"))
    if os.path.exists(vec_path) and os.path.getsize(vec_path) > 0:
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


# --------- FIXED ROUTE (POST JSON + GET EASY SUPPORT) ---------
@app.route("/get", methods=["GET", "POST"])
def get_response():
    if request.method == "POST":
        # JSON body support
        data_json = request.get_json()
        msg = data_json.get("msg") if data_json else None
    else:
        # GET fallback support
        msg = request.args.get("msg")

    if not msg:
        return jsonify({"response": "Please type something!"})

    reply = chatbot_response(msg)
    return jsonify({"response": reply})


# --------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
