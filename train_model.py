import json, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

data = json.load(open("data/intents.json"))

patterns = []
labels = []

for intent in data["intents"]:
    for p in intent["patterns"]:
        patterns.append(p)
        labels.append(intent["intent"])

vec = TfidfVectorizer()
X = vec.fit_transform(patterns)

model = LogisticRegression(max_iter=200)
model.fit(X, labels)

pickle.dump(model, open("model/intent_model.pkl","wb"))
pickle.dump(vec, open("model/vectorizer.pkl","wb"))

print("Training Completed ✔")
