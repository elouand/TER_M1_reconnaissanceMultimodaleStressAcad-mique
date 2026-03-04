import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from collections import deque
import pyaudio # Pour l'audio futur
from traitementVideo import EmotionRegressor

# --- CONFIGURATION ÉMULATEUR ---
WIDTH, HEIGHT = 640, 480 
RATE = 48000
CHUNK = 1024
MODEL_PATH = 'detector.tflite'
SEQ_LEN = 10 # On augmente le lissage à cause du bruit de l'image

# 1. INITIALISATION (MediaPipe + Emotion)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.FaceDetectorOptions(base_options=base_options)
detector = vision.FaceDetector.create_from_options(options)
regressor = EmotionRegressor()

# 2. INITIALISATION AUDIO (Préparation Wav2Vec2)
p = pyaudio.PyAudio()
audio_stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Buffers de lissage
v_buffer = deque(maxlen=SEQ_LEN)
a_buffer = deque(maxlen=SEQ_LEN)

cap = cv2.VideoCapture(0)

print("--- Système Pepper Actif (Vidéo Dégradée + Audio Ready) ---")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    # --- SIMULATION PROXY PEPPER ---
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    # Ajout du bruit (grain numérique)
    noise = np.random.randint(0, 30, (HEIGHT, WIDTH, 3), dtype='uint8')
    frame = cv2.add(frame, noise)
    # Compression JPEG (artefacts)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
    _, encimg = cv2.imencode('.jpg', frame, encode_param)
    frame = cv2.imdecode(encimg, 1)

    # --- TRAITEMENT AUDIO (Prélude Wav2Vec2) ---
    audio_data = audio_stream.read(CHUNK, exception_on_overflow=False)
    # Ici, tu pourrais envoyer audio_data à une fonction Wav2Vec2

    # --- ANALYSE D'ÉMOTION ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    detection_result = detector.detect(mp_image)

    if detection_result.detections:
        bbox = detection_result.detections[0].bounding_box
        x, y, w, h = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height
        
        # Crop sécurisé
        face_img = rgb_frame[max(0,y):y+h, max(0,x):x+w]
        
        if face_img.size > 0:
            v, a = regressor.get_va(face_img)
            v_buffer.append(v)
            a_buffer.append(a)
            
            v_smooth = sum(v_buffer) / len(v_buffer)
            a_smooth = sum(a_buffer) / len(a_buffer)

            # --- Rendu visuel ---
            color = (0, 255, 0) if v_smooth > 0 else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, f"V: {v_smooth:.2f} A: {a_smooth:.2f}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, "FLUX PEPPER SIMULATED", (10, HEIGHT - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.imshow('TER Pepper - Reconnaissance Multimodale', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Nettoyage
cap.release()
cv2.destroyAllWindows()
audio_stream.stop_stream()
p.terminate()