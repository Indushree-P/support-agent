from flask import Flask, request, render_template, session, redirect, url_for
import pickle
import re
from flask_session import Session

app = Flask(__name__)
app.secret_key = "chat_secret_key"
app.config["SESSION_TYPE"] = "filesystem"
Session(app)

# Load the trained ML model
model = pickle.load(open("chatbot_model.pkl", "rb"))

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Define responses
responses = {
    "order_status": "You can track your order using the 'My Orders' section.",
    "returns": "You can return or exchange an item within 30 days.",
    "refund": "Refunds are processed in 5-7 business days.",
    "cancel_order": "You can cancel an order within 12 hours of purchase.",
    "exchange": "You can exchange items at our service centers.",
    "payment_methods": "Yes, it is possible to pay through those methods.",
    "app_issue": "Check your connection and retry.",
    "general_help": "Sure! Tell me how I can help you.",
    "warranty": "Warranty usually depends on the product, but all items have a one-year warranty.",
    "product_suggestion": "Sure! Specify the product you want to purchase.",
    "product_availability": "Yes, we have these products available.",
}

# Home route
@app.route("/", methods=["GET", "POST"])
def home():
    if "chat_history" not in session:
        session["chat_history"] = []

    if request.method == "POST":
        user_input = request.form["msg"]
        clean_input = preprocess(user_input)

        # Predict the intent using the ML model
        intent = model.predict([clean_input])[0]

        # Get the response based on intent
        bot_reply = responses.get(intent, "Sorry, I didn’t understand that.")

        # Save to session history
        session["chat_history"].append({"user": user_input, "bot": bot_reply})
        session.modified = True

    return render_template("index.html", chat_history=session["chat_history"])

# Clear chat
@app.route("/clear")
def clear():
    session.clear()
    return redirect(url_for("home"))

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True) 

   