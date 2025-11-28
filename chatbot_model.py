import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import pickle
import re
import matplotlib.pyplot as plt  # <- Add this import

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    return text

# Load and preprocess data
df = pd.read_csv("chat_data.csv")
df["text"] = df["text"].apply(preprocess)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# Train model
model = make_pipeline(TfidfVectorizer(), LogisticRegression(max_iter=1000))
model.fit(X_train, y_train)

# Save model
with open("chatbot_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Evaluate model
accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Plot Confusion Matrix
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
ConfusionMatrixDisplay(cm, display_labels=model.classes_).plot()
plt.xticks(rotation=45)
plt.tight_layout() 
plt.show()  