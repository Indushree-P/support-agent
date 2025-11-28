# Customer Support Chatbot

Lightweight Machine Learning–based Support Chatbot built for a 48-hour AI Agent Development Challenge.

## What it does
- Uses a trained ML model (`chatbot_model.pkl`) to classify user queries
- Answers common support FAQs (refund, returns, warranty, order status, payments, etc.)
- Supports conversation flow with chat history stored in session
- Provides a clean animated UI with dark mode
- Runs fully offline — no external APIs required

## Files
- `app.py` — Flask backend & chatbot logic
- `chatbot_model.py` — ML training script (TF-IDF + Logistic Regression)
- `chatbot_model.pkl` — Saved trained model
- `chat_data.csv` — Dataset used for training intents
- `templates/index.html` — Frontend chat UI
- `architecture.png` — Architecture diagram (add here)
- `README.md` — Documentation

## How to run (local)
1. Install dependencies:
   ```bash
   pip install flask flask-session scikit-learn pandas
   ```

2. (Optional) Retrain the model if you updated `chat_data.csv`:
   ```bash
   python chatbot_model.py
   ```

3. Run the Flask app:
   ```bash
   python app.py
   ```

4. Open in browser:
   ```
   http://127.0.0.1:5000
   ```

## Notes & Improvements
- Currently answers only predefined intents
- Add embeddings/LLMs to handle open-ended queries
- Add database to store real log history
- Add voice input / multilingual support
- Integrate real product/order APIs for live data

## Submission checklist for challenge
- Working Demo Link (local Flask or hosted)
- GitHub repo with source code & README
- Architecture diagram (`architecture.png`) included
- Optional 2–3 minute demo video

Good luck! 🚀💬
