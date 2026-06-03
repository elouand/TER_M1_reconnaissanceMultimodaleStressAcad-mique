import cv2
import numpy as np
import pyaudio
import socket

# --- CONFIGURATION RÉSEAU ---
TARGET_IP = "127.0.0.1"
PORT_VIDEO = 5005
PORT_AUDIO = 5006

# --- CONFIGURATION AUDIO/VIDÉO ---
RATE = 48000
CHUNK = 1024 
WIDTH, HEIGHT = 640, 480 

FPS_CIBLE = 5  # Nombre d'images par seconde souhaité
DELAY = int(1000 / FPS_CIBLE)

# Initialisation Sockets
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Initialisation Audio
p = pyaudio.PyAudio()
audio_stream = p.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Initialisation Vidéo (Webcam)
cap = cv2.VideoCapture(0)

print("--- Émulateur Pepper (Mode Silencieux) ---")
print("Envoi UDP en cours... (Ctrl+C pour arrêter)")

try:
    while True:
        # 1. TRAITEMENT VIDÉO
        ret, frame = cap.read() # Capture nativement en BGR
        if not ret: break
        
        frame = cv2.resize(frame, (WIDTH, HEIGHT))

        frame_pepper_format = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Simulation du bruit de capteur Pepper
        noise = np.random.randint(0, 15, (HEIGHT, WIDTH, 3), dtype='uint8')
        frame = cv2.add(frame_pepper_format, noise)
        
        # Encodage JPEG (C'est ce que reçoit le imdecode du receveur)
        # On règle la qualité à 30 pour simuler la compression réseau du robot
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]
        _, img_encoded = cv2.imencode('.jpg', frame, encode_param)
        
        # Envoi de la trame compressée
        sock.sendto(img_encoded.tobytes(), (TARGET_IP, PORT_VIDEO))

        # 2. TRAITEMENT AUDIO
        try:
            # Lecture du micro PC (Mono)
            audio_raw = audio_stream.read(CHUNK, exception_on_overflow=False)
            audio_np = np.frombuffer(audio_raw, dtype=np.int16)
            
            # Simulation de l'entrelacement 4 canaux de Pepper
            # On répète chaque échantillon 4 fois : [A, A, A, A, B, B, B, B...]
            four_channels = np.repeat(audio_np, 4)
            
            # Ajout du sifflement des ventilateurs (Bruit Gaussien)
            #fan_noise = np.random.normal(0, 300, len(four_channels)).astype(np.int16)
            combined = four_channels #+ fan_noise
            
            sock.sendto(combined.tobytes(), (TARGET_IP, PORT_AUDIO))
        except Exception as e:
            pass

        if cv2.waitKey(DELAY) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\nArrêt de l'émulateur.")

finally:
    cap.release()
    audio_stream.stop_stream()
    audio_stream.close()
    p.terminate()
    sock.close()