import socket
import cv2
import numpy as np
import threading
import pyaudio
from scipy.signal import butter, lfilter

# Config Réseau
UDP_IP = "0.0.0.0"
PORT_VIDEO = 5005
PORT_AUDIO = 5006

# Config Audio (PyAudio pour lecture en temps réel)
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)

def butter_highpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a

def apply_cleaning(data, fs=48000):
    # 1. Conversion en float pour le calcul
    audio_float = data.astype(np.float32)
    
    # 2. Filtre Passe-Haut (on coupe tout sous 250Hz - le "vroum" du moteur)
    b, a = butter_highpass(250, fs, order=5)
    filtered = lfilter(b, a, audio_float)
    
    # 3. Noise Gate "Soft" (on réduit le volume des sons très faibles)
    # Si l'amplitude est sous un seuil, on l'écrase
    threshold = np.max(np.abs(filtered)) * 0.05
    filtered[np.abs(filtered) < threshold] = 0
    
    # 4. Normalisation automatique du gain
    # Ça permet d'avoir toujours le même niveau sonore pour l'IA
    if np.max(np.abs(filtered)) > 0:
        filtered = filtered / np.max(np.abs(filtered))
        
    return (filtered * 32767).astype(np.int16)

# --- DANS TON AUDIO_THREAD ---
def audio_thread():
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_audio.bind((UDP_IP, PORT_AUDIO))
    
    # Paramètres du Gate
    THRESHOLD = 380 # Augmente cette valeur pour couper plus de bruit
    
    print("Écoute Audio active...")
    while True:
        try:
            data, addr = sock_audio.recvfrom(65535)
            # 1. Extraction Mono
            audio_data = np.frombuffer(data, dtype=np.int16)[0::4]
            
            # 2. Noise Gate simple mais efficace
            # On calcule le volume moyen du paquet
            volume = np.abs(audio_data).mean()
            
            if volume < THRESHOLD:
                # Si c'est juste le bruit du ventilo, on met du silence
                clean_audio = np.zeros_like(audio_data)
            else:
                # Sinon on laisse passer et on booste un peu la voix
                clean_audio = audio_data * 1.5 
                # On limite pour éviter de saturer (clipping)
                clean_audio = np.clip(clean_audio, -32768, 32767)

            # 3. Envoi aux enceintes
            stream.write(clean_audio.astype(np.int16).tobytes())
            
        except Exception as e:
            print(f"Erreur Audio: {e}")

# Lancement du thread Audio
t_audio = threading.Thread(target=audio_thread)
t_audio.daemon = True
t_audio.start()

# --- BOUCLE PRINCIPALE (VIDÉO) ---
sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_video.bind((UDP_IP, PORT_VIDEO))

print("Réception Vidéo et Audio en cours...")

try:
    while True:
        # 1. Recevoir l'image
        data_v, addr_v = sock_video.recvfrom(65535)
        nparr = np.frombuffer(data_v, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
        
        if frame is not None:
            # 2. Affichage
            cv2.imshow("Flux Pepper Synchro", frame)
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    stream.stop_stream()
    p.terminate()