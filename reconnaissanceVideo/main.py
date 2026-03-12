import cv2
import numpy as np
from collections import deque
from traitementVideo import get_visual_vad

# --- CONFIGURATION ---
SEQ_LEN = 5
v_buffer = deque(maxlen=SEQ_LEN)
a_buffer = deque(maxlen=SEQ_LEN)
d_buffer = deque(maxlen=SEQ_LEN)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    
    # Effet miroir et détection
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))

    for (x, y, w, h) in faces:
        # Extraction du visage pour l'IA
        face_img = frame[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        
        # Récupération VAD avec TA logique (Inversion + Tanh)
        v, a, d = get_visual_vad(face_rgb)
        
        # Lissage temporel
        v_buffer.append(v); a_buffer.append(a); d_buffer.append(d)
        v_smooth = sum(v_buffer) / len(v_buffer)
        a_smooth = sum(a_buffer) / len(a_buffer)
        d_smooth = sum(d_buffer) / len(d_buffer)

        # Affichage (Dashboard Pepper)
        color = (0, 255, 0) if v_smooth > 0 else (0, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        cv2.putText(frame, f"V: {v_smooth:+.2f}", (x, y-55), 1, 1.2, (255,255,255), 2)
        cv2.putText(frame, f"A: {a_smooth:+.2f}", (x, y-30), 1, 1.2, (255,255,255), 2)
        cv2.putText(frame, f"D: {d_smooth:+.2f}", (x, y-5), 1, 1.2, (255,255,255), 2)
        break 

    cv2.imshow('Pepper - VAD Logic Protected', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows() 