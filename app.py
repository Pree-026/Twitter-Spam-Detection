from flask import Flask, render_template, request, jsonify
import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Initialize Flask app
app = Flask(__name__)

# Try loading the model and vectorizer
try:
    model = joblib.load("naive_bayes_model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("✅ Model and vectorizer loaded.")
except Exception as e:
    print("❌ Error loading model/vectorizer:", e)

# Download required NLTK data
nltk.download('stopwords')
nltk.download('wordnet')

# Set up text processing tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

# Route for frontend
@app.route("/")
def index():
    return render_template("index.html")

# Route for spam prediction
@app.route("/predict", methods=["POST"])
def predict():
    if request.is_json:
        data = request.get_json()
        tweet = data.get("tweet", "")
        print("🔍 Received tweet:", tweet)

        try:
            cleaned = clean_text(tweet)
            print("🧹 Cleaned tweet:", cleaned)

            vect = vectorizer.transform([cleaned])
            prediction = model.predict(vect)[0]
            print("🤖 Prediction result:", prediction)

            result = "Spam" if prediction else "Ham"
            return jsonify({"prediction": result})

        except Exception as e:
            print("❌ Error during prediction:", str(e))
            return jsonify({"error": str(e)}), 500

    else:
        print("❌ Invalid content type, expecting application/json.")
        return jsonify({"error": "Unsupported Media Type"}), 415

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
