import pickle
import re
import pandas as pd

# Load the saved Random Forest model and vectorizer
with open('../model/random_forest_model.pkl', 'rb') as rf_model_file:
    rf_model = pickle.load(rf_model_file)

with open('../model/vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load the dataset with company profiles
df = pd.read_csv('/mnt/c/Users/nikhi/Downloads/ai/Fake Postings.csv')

# Preprocess function: lower and strip
def preprocess_text(text):
    return re.sub(r'\s+', ' ', str(text).lower().strip())

# Create a set of preprocessed company profiles for exact matching
company_profiles_set = set(df['company_profile'].dropna().map(preprocess_text))

print("Enter company profiles one per line. When done, enter an empty line:")

real_count = 0
fake_count = 0

while True:
    user_input = input().strip()
    if user_input == '':
        break

    preprocessed_input = preprocess_text(user_input)

    print(f"\nCompany Profile: {user_input}")

    # Rule: If single character or empty input => automatically fake
    if len(preprocessed_input) <= 1:
        print("Status:  Input too short or single character — Marked as Fake")
        fake_count += 1
        print("-" * 50)
        continue

    # Check exact match
    if preprocessed_input in company_profiles_set:
        print("Status:  Present in dataset — Marked as Real")
        real_count += 1
    else:
        print("Status:  Not present in dataset — Marked as Fake")
        fake_count += 1

    # Predict with Random Forest anyway
    X_new = vectorizer.transform([preprocessed_input])
    rf_prob = rf_model.predict_proba(X_new)[0][1]
    rf_pred = rf_model.predict(X_new)[0]

    print(f"Random Forest Fraud Probability: {rf_prob:.3f}")
    print(f"Random Forest Model Prediction: {'Fraudulent' if rf_pred == 1 else 'Real'}")

    print("-" * 50)

print(f"\nSummary:")
print(f"Total Real Companies: {real_count}")
print(f"Total Fake Companies: {fake_count}")
