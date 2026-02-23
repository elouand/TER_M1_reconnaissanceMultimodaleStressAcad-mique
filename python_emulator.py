import cv2
import numpy as np
import pyaudio
import wave

# --- CONFIGURATION ---
RATE = 48000
CHUNK = 1024
# Simulation de la résolution Pepper (640x480 est le standard)
WIDTH, HEIGHT = 640, 480 

# Initialisation Audio
p = pyaudio.PyAudio()
audio_stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialisation Vidéo
cap = cv2.VideoCapture(0)

print("--- Émulateur Pepper (Audio + Vidéo) ---")
print("Appuie sur 'q' pour arrêter l'enregistrement.")

frames_audio = []

while True:
    # 1. CAPTURE VIDÉO + DÉGRADATION
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.resize(frame, (WIDTH, HEIGHT))
    
    # Simulation du bruit numérique (grain)
    noise = np.random.randint(0, 30, (HEIGHT, WIDTH, 3), dtype='uint8')
    frame = cv2.add(frame, noise)
    
    # Simulation de la compression JPEG (artefacts)
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30] # Qualité basse
    result, encimg = cv2.imencode('.jpg', frame, encode_param)
    frame = cv2.imdecode(encimg, 1)

    # 2. CAPTURE AUDIO + BRUIT VENTILO
    audio_data = audio_stream.read(CHUNK)
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32)
    # Ajout d'un sifflement constant (ventilo)
    fan_noise = np.random.normal(0, 500, CHUNK).astype(np.float32)
    combined = (audio_np + fan_noise).astype(np.int16)
    frames_audio.append(combined.tobytes())

    # Affichage
    cv2.putText(frame, "SIMULATION PEPPER", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Rendu IA (Pepper Cam)', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Sauvegarde finale
with wave.open("dataset_test.wav", "wb") as wf:
    wf.setnchannels(1)
    wf.setsampwidth(2)
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames_audio))

cap.release()
cv2.destroyAllWindows()
audio_stream.stop_stream()
p.terminate()