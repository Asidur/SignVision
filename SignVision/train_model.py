import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('sign_data.csv', header=None)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Restrict to valid labels
valid_labels = [chr(i) for i in range(65, 91) if chr(i) not in ['J', 'Z']]
mask = np.array([lbl in valid_labels for lbl in y])
X_scaled, y = X_scaled[mask], y[mask]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=40, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Validation Accuracy:", accuracy_score(y_test, y_pred))

with open('sign_model.pkl', 'wb') as f:
    pickle.dump(clf, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Model/scaler saved!")
