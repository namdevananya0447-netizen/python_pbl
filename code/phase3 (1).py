import pickle
with open('best_model.pkl', 'rb') as f:
    loaded_model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    loaded_scaler = pickle.load(f)

print("Model loaded successfully.")
import pandas as pd

# Feature names (same as training)
feature_names = [
    '_BMI5', 'BPHIGH4', 'BLOODCHO', 'SMOKE100', 'EXERANY2', 
    'GENHLTH', 'MENTHLTH', 'PHYSHLTH', 'SEX', 'EDUCA', 'INCOME2'
]

print("Enter Patient Details:\n")

bmi = float(input("Enter BMI: "))
bp = int(input("High BP (1=Yes, 0=No): "))
chol = int(input("High Cholesterol (1=Yes, 0=No): "))
smoke = int(input("Smoked 100 cigarettes (1=Yes, 0=No): "))
exercise = int(input("Exercise (1=Yes, 0=No): "))
genhlth = int(input("General Health (1-5): "))
menthlth = int(input("Mental Health Days (0-30): "))
physhlth = int(input("Physical Health Days (0-30): "))
sex = int(input("Sex (1=Male, 0=Female): "))
educa = int(input("Education (1-6): "))
income = int(input("Income (1-8): "))

# Create DataFrame
user_df = pd.DataFrame([[bmi, bp, chol, smoke, exercise, genhlth,
                         menthlth, physhlth, sex, educa, income]],
                       columns=feature_names)

print("\nInput captured successfully!")
import numpy as np

# Scale input
user_scaled = loaded_scaler.transform(user_df)

# Prediction
prediction = loaded_model.predict(user_scaled)

# Confidence handling
if hasattr(loaded_model, "predict_proba"):
    prob = loaded_model.predict_proba(user_scaled)
    confidence = np.max(prob) * 100
else:
    score = loaded_model.decision_function(user_scaled)
    confidence = (1 / (1 + np.exp(-score)))[0] * 100  # sigmoid approx

# Output result
print("\n" + "="*40)
if prediction[0] == 1:
    print("Prediction: DIABETIC")
else:
    print("Prediction: NOT DIABETIC")

print(f"Confidence Level: {confidence:.2f}%")
print("="*40)