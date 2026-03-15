import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

data = pd.read_csv('dataset_isyarat.csv')

x = data.drop('label', axis = 1)
y = data['label']

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
print("--- Sedang Melatih Otak AI ---")

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
score = accuracy_score(y_test, y_pred)

print(f"Selesai! Akurasi AI kamu: {score * 100:.2f}%")

with open('model_gesture.pkl', 'wb') as f:
    pickle.dump(model, f)

print("File 'model_gesture.pkl' berhasil di buat!")