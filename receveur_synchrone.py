import socket
import cv2
import numpy as np
import threading
import pyaudio
import time
import os
from ctypes import *

from vidToVAD import get_visual_vad
from audToVAD import get_acoustic_vad
from MultimodalState import MultimodalState 

# --- 1. SUPPRESSION DES LOGS ALSA (VERSION SÉCURISÉE) ---
def py_error_handler(filename, line, function, err, fmt):
    pass # Ne fait rien, ignore l'erreur

def silence_alsa():
    # On définit le type de callback
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    # On stocke le handler dans une variable globale pour éviter qu'il soit nettoyé par le Garbage Collector
    global c_error_handler
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    
    try:
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
    except Exception as e:
        print(f"Note: Impossible de silencer ALSA ({e})")

# Appeler le silence AVANT toute initialisation audio
silence_alsa()

# --- INITIALISATION ---
UDP_IP = "0.0.0.0"
PORT_VIDEO, PORT_AUDIO = 5005, 5006
audio_buffer = []
state_manager = MultimodalState()

# --- THREAD AUDIO ---
def run_audio_listening():
    # Initialisation locale de PyAudio
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)
    
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_audio.bind((UDP_IP, PORT_AUDIO))
    
    global audio_buffer
    print("Écoute audio activée...")
    while True:
        try:
            data, _ = sock_audio.recvfrom(8192)
            raw = np.frombuffer(data, dtype=np.int16)
            audio_mono = raw[0::4]
            
            audio_stream.write(audio_mono.tobytes())
            
            audio_buffer.append(audio_mono)
            if len(audio_buffer) >= 45: # ~1s
                segment = np.concatenate(audio_buffer).astype(np.float32) / 32768.0
                # On lance l'analyse dans un thread séparé
                threading.Thread(target=audio_analysis_task, args=(segment,), daemon=True).start()
                audio_buffer = []
        except:
            pass

def audio_analysis_task(segment):
    # Calcul rapide du volume (RMS) pour savoir si c'est du silence
    rms = np.sqrt(np.mean(segment**2))
    
    # Seuil empirique : n'envoie à la matrice que si on parle vraiment
    if rms > 0.02: 
        res = get_acoustic_vad(segment, sampling_rate=48000)
        if res:
            v = (res['valence'] * 2) - 1
            a = (res['arousal'] * 2) - 1
            d = (res['dominance'] * 2) - 1
            
            # On ne met à jour la matrice QUE s'il y a du son
            state_manager.update("audio", [v, a, d])
    else:
        # Optionnel : on ne fait rien. 
        # La valeur audio dans la matrice va s'amortir toute seule 
        # car son timestamp ne sera pas mis à jour.
        pass

# --- BOUCLE PRINCIPALE ---
def main():
    # Démarrage du thread audio
    t_audio = threading.Thread(target=run_audio_listening, daemon=True)
    t_audio.start()
    
    sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_video.bind((UDP_IP, PORT_VIDEO))
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    print("--- Système Multimodal Prêt ---")

    try:
        while True:
            data_v, _ = sock_video.recvfrom(65535)
            nparr = np.frombuffer(data_v, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                # Correction Pepper
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # IA Vision
                v_v, a_v, d_v = get_visual_vad(frame)
                state_manager.update("vision", [v_v, a_v, d_v])

                # Fusion
                v_f, a_f, d_f = state_manager.get_fusion()

                # Dessin
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(100, 100))
                
                for (x, y, w, h) in faces:
                    color = (0, 255, 0) if v_f >= 0 else (0, 0, 255)
                    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                    label = f"V:{v_f:.2f} A:{a_f:.2f} D:{d_f:.2f}"
                    cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    break
                
                cv2.imshow("Analyse Multimodale", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()