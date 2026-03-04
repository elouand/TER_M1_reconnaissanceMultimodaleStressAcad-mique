import socket
import cv2
import numpy as np
import threading
import pyaudio
from scipy.signal import butter, lfilter
from vidToVAD import get_visual_vad
from audToVAD import get_acoustic_vad

# Config Réseau
UDP_IP = "0.0.0.0"
PORT_VIDEO = 5005
PORT_AUDIO = 5006

# Config Audio (PyAudio pour lecture en temps réel)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)
CHUNKS_AUDIO = 45 # Environ 1 seconde de son à 48kHz (si chunk=1024)
audio_buffer = []

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def audio_thread():
    global audio_buffer
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_audio.bind((UDP_IP, PORT_AUDIO))
    
    while True:
        try:
            # 1. On augmente le buffer à 8192 octets (la taille exacte envoyée par l'émulateur)
            # On peut même mettre 65535 (le max UDP) pour être totalement serein
            data, addr = sock_audio.recvfrom(8192) 
            audio_data = np.frombuffer(data, dtype=np.int16)
            
            # 2. Séparation des canaux : on ne garde que le 1er micro sur les 4
            # C'est vital pour que l'IA entende une voix humaine normale
            audio_mono = audio_data[0::4]
            
            # 3. Lecture directe pour le monitoring sur les enceintes du PC
            stream.write(audio_mono.tobytes())

            # 4. Accumulation pour l'analyse VAD
            audio_buffer.append(audio_mono)
            
            if len(audio_buffer) >= CHUNKS_AUDIO:
                # On fusionne les chunks (environ 1 seconde d'audio)
                segment = np.concatenate(audio_buffer).astype(np.float32) / 32768.0
                
                # Inférence dans un thread séparé
                threading.Thread(target=audio_analysis, args=(segment,)).start()
                
                audio_buffer = [] 

        except Exception as e:
            print(f"Erreur thread audio : {e}")
            
def audio_analysis(segment):
    # Appel à audToVAD qui renvoie désormais le dictionnaire complet {arousal, dominance, valence}
    scores = get_acoustic_vad(segment, sampling_rate=48000)
    
    if scores:
        # On ajoute dominance ici pour l'affichage
        print(f"TON : Valence={scores['valence']:.2f}, "
              f"Arousal={scores['arousal']:.2f}, "
              f"Dominance={scores['dominance']:.2f}")

# Lancement du thread Audio
t_audio = threading.Thread(target=audio_thread)
t_audio.daemon = True
t_audio.start()

# --- BOUCLE PRINCIPALE (VIDÉO + IA) ---
sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_video.bind((UDP_IP, PORT_VIDEO))

# Détecteur de visage pour le dessin du rectangle (OpenCV local)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

print("Réception Vidéo/Audio + IA VAD activée...")

try:
    while True:
        data_v, addr_v = sock_video.recvfrom(65535)
        nparr = np.frombuffer(data_v, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is not None:
            # 1. Correction du format de couleur (Pepper RGB -> PC BGR)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # 2. Appel de l'IA (Vision)
            # Ta fonction get_visual_vad renvoie np.array([v, a, d])
            v, a, d = get_visual_vad(frame)
            
            # 3. Détection locale pour le dessin du rectangle
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
            
            for (x, y, w, h) in faces:
                # Couleur dynamique selon la Valence (Vert si positif, Rouge si négatif)
                color = (0, 255, 0) if v >= 0 else (0, 0, 255)
                
                # Dessin du rectangle et texte VAD
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                text = f"V: {v:.2f} A: {a:.2f} D: {d:.2f}"
                cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                break # On ne traite que le premier visage pour la fluidité
            
            # 4. Affichage final
            cv2.imshow("Flux Pepper Synchro + IA VAD", frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    stream.stop_stream()
    p.terminate()