import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import speech_recognition as sr
import subprocess

def reconocer_gesto_por_voz():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("🎙️ Diga el nombre del gesto...")
        audio = recognizer.listen(source)
        try:
            gesto = recognizer.recognize_google(audio, language="es-ES").lower()
            if "letra" in gesto:
                gesto = gesto.replace("letra ", "")  # Elimina "letra "
            print(f"✅ Gesto detectado: {gesto}")
            return gesto
        except sr.UnknownValueError:
            print("❌ No se entendió el gesto, intente de nuevo.")
            return None

# Inicializar MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# 🔹 Pedir al usuario el gesto que quiere capturar por voz
GESTURE_NAME = None
while GESTURE_NAME is None:
    GESTURE_NAME = reconocer_gesto_por_voz()

# Archivo donde guardaremos los datos
CSV_FILE = "gestos_dataset.csv"

cap = cv2.VideoCapture(0)

data = []

print(f"📸 Mostrando gesto: {GESTURE_NAME}. Pulsa 's' para guardar una muestra, 'q' para salir.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extraer coordenadas
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            # Convertir a un array plano
            landmarks = np.array(landmarks).flatten()

            # Guardar datos si el usuario presiona 's'
            if cv2.waitKey(1) & 0xFF == ord('s'):
                data.append(np.append(landmarks, GESTURE_NAME))
                print(f"✅ Muestra guardada para {GESTURE_NAME}")

    cv2.imshow("Captura de Gestos", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty("Captura de Gestos", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()

# Guardar datos en CSV
df = pd.DataFrame(data)
df.to_csv(CSV_FILE, mode='a', header=False, index=False)
print(f"📁 Datos guardados en {CSV_FILE}")
print("🔄 Ejecutando entrenar_modelo.py")
ruta_script = r"C:\Users\valen\OneDrive\Escritorio\Fer\projects\face_recognition_project\entrenar_modelo.py"
subprocess.run(["python", ruta_script])