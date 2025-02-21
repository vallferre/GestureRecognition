import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Cargar los datos
CSV_FILE = "gestos_dataset.csv"
df = pd.read_csv(CSV_FILE, header=None)

# Separar características (X) y etiquetas (y)
X = df.iloc[:, :-1].values  # Coordenadas de los puntos de la mano
y = df.iloc[:, -1].values   # Etiquetas de los gestos

# Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entrenar modelo
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Guardar el modelo
joblib.dump(model, "modelo_gestos.pkl")
print("✅ Modelo entrenado y guardado como modelo_gestos.pkl")

# Evaluar precisión
accuracy = model.score(X_test, y_test)
print(f"🎯 Precisión del modelo: {accuracy * 100:.2f}%")