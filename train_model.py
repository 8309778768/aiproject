import pandas as pd
import numpy as np
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import os
import random

# === Utility ===
def clean_text(text):
    if pd.isna(text):
        return ""
    return re.sub(r'\W+', ' ', text).lower().strip()

# === Load dataset ===
df = pd.read_csv('/mnt/c/Users/nikhi/Downloads/ai/Fake Postings.csv')  # adjust path if needed
df['company_profile_clean'] = df['company_profile'].apply(clean_text)
df = df[df['company_profile_clean'] != ""]

# === Prepare real samples ===
real_df = df[['company_profile_clean']].drop_duplicates().sample(n=300, random_state=42)
real_df['label'] = 0

# === Create synthetic fake samples ===
fake_profiles = [
    f"this is a fake company profile number {i} promising fast income from home"
    for i in range(300)
]
fake_df = pd.DataFrame({'company_profile_clean': fake_profiles, 'label': 1})

# === Combine both ===
final_df = pd.concat([real_df, fake_df], ignore_index=True)
print("✅ Final class distribution:\n", final_df['label'].value_counts())

# === TF-IDF vectorization ===
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(final_df['company_profile_clean'])
y = final_df['label']

# === Train/test split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# === Train Logistic Regression ===
lr = LogisticRegression(max_iter=300)
lr.fit(X_train, y_train)

# === Train Random Forest ===
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# === Evaluate ===
def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

evaluate_model("Logistic Regression", lr, X_test, y_test)
evaluate_model("Random Forest", rf, X_test, y_test)

# === Save models and vectorizer ===
os.makedirs("model", exist_ok=True)
pickle.dump(lr, open("model/logistic_model.pkl", "wb"))
pickle.dump(rf, open("model/random_forest_model.pkl", "wb"))
pickle.dump(vectorizer, open("model/vectorizer.pkl", "wb"))
pickle.dump(set(real_df['company_profile_clean']), open("model/known_companies.pkl", "wb"))

# === Test on a new profile ===
def predict_company_profile(text):
    text_clean = clean_text(text)
    print(f"\n🔍 Testing: {text_clean}")
    vector = vectorizer.transform([text_clean])
    pred_lr = lr.predict(vector)[0]
    pred_rf = rf.predict(vector)[0]

    print(f"🔹 Logistic Regression: {'Fake' if pred_lr == 1 else 'Real'}")
    print(f"🔹 Random Forest:      {'Fake' if pred_rf == 1 else 'Real'}")

# === Example Prediction ===
predict_company_profile("guaranteed passive income from home today")
