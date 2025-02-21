import cv2
import mediapipe as mp
import numpy as np
import joblib
import time

TIEMPO_MAX = 2  # Segundos sin nueva detección para considerar palabra completa

letras_detectadas = []  # Almacena las letras detectadas
ultima_letra = None  # Última letra detectada para evitar repeticiones
ultimo_tiempo = time.time()  # Tiempo de la última detección

def agregar_letra(letra):
    """Agrega una letra si es diferente a la última detectada y reinicia el temporizador."""
    global ultima_letra, ultimo_tiempo
    if letra != ultima_letra:  # Evita letras repetidas consecutivas
        letras_detectadas.append(letra)
        ultima_letra = letra
        ultimo_tiempo = time.time()

def obtener_palabra():
    """Si ha pasado suficiente tiempo, devuelve la palabra y reinicia la lista."""
    global letras_detectadas, ultima_letra
    if time.time() - ultimo_tiempo > TIEMPO_MAX:
        palabra = "".join(letras_detectadas)
        letras_detectadas.clear()  # Reiniciar lista para la siguiente palabra
        ultima_letra = None  # Resetear última letra detectada
        return palabra if palabra.strip() else None
    return None

modelo = joblib.load("modelo_gestos.pkl")

mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
print("🎥 Cámara encendida. Realiza un gesto para reconocerlo.")

with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convertir a RGB para MediaPipe
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Procesar detección de rostros y manos
        face_results = face_detection.process(frame_rgb)
        hand_results = hands.process(frame_rgb)
        
        # Dibujar rostros detectados
        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(frame, detection)
        
        # Procesar detección de manos y predecir gestos
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extraer coordenadas normalizadas de los 21 puntos de la mano
                landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten().reshape(1, -1)
                
                # Intentar predecir el gesto
                try:
                    prediction = modelo.predict(landmarks)[0]
                    agregar_letra(prediction)
                except Exception:
                    prediction = "Desconocido"
                
                # Mostrar el gesto detectado
                cv2.putText(frame, f"Gesto: {prediction}", (50, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Verificar si se ha formado una palabra completa
        palabra_formada = obtener_palabra()
        if palabra_formada:
            print(f"🔤 Palabra detectada: {palabra_formada}")
            cv2.putText(frame, f"Palabra: {palabra_formada}", (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        
        # Mostrar la imagen con las detecciones
        cv2.imshow('Reconocimiento de Gestos y Caras', frame)
        
        # Salir con 'q' o si la ventana es cerrada
        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.getWindowProperty('Reconocimiento de Gestos y Caras', cv2.WND_PROP_VISIBLE) < 1:
            break

cap.release()
cv2.destroyAllWindows()
print("🛑 Programa cerrado.")