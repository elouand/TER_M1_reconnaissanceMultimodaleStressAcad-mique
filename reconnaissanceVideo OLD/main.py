import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
from traitementVideo import EmotionRegressor

# --- CONFIGURATION ---
MODEL_PATH = 'detector.tflite'
SEQ_LEN = 5 # On garde le buffer pour lisser les résultats (optionnel mais recommandé)

# 1. Initialisation de MediaPipe Tasks
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)

# 2. Initialisation de notre nouveau Regresseur
regressor = EmotionRegressor()

# Buffer pour lisser les scores (évite que les chiffres sautent trop)
v_buffer = deque(maxlen=SEQ_LEN)
a_buffer = deque(maxlen=SEQ_LEN)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # MediaPipe a besoin de RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # Détection
    detection_result = detector.detect(mp_image)

    if detection_result.detections:
        detection = detection_result.detections[0]
        bbox = detection.bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

        # Extraction du visage (avec protection des bords)
        face_img = rgb_frame[max(0,y):y+h, max(0,x):x+w]
        
        if face_img.size > 0:
            # Récupération des scores VA directs
            v, a = regressor.get_va(face_img)
            
            v_buffer.append(v)
            a_buffer.append(a)
            
            # Moyenne glissante pour la stabilité
            v_smooth = sum(v_buffer) / len(v_buffer)
            a_smooth = sum(a_buffer) / len(a_buffer)

            # Affichage dynamique
            # Vert si positif, Rouge si négatif
            color = (0, 255, 0) if v_smooth > 0 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            
            text = f"Valence: {v_smooth:.2f} | Arousal: {a_smooth:.2f}"
            cv2.putText(frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imshow('Pepper - Emotion VAD Unimodal', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()