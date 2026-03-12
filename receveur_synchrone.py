import socket
import cv2
import numpy as np
import threading
import pyaudio
import time
import signal
from ctypes import *

from vidToVAD import get_visual_vad
from audToVAD import get_acoustic_vad, get_text_vad
from MultimodalState import MultimodalState 

# --- VERROUS DE SÉCURITÉ (Évite les crashs CPU 0xC0000005) ---
lock_ton = threading.Lock()
lock_texte = threading.Lock()

# --- SUPPRESSION DES LOGS ALSA ---
def py_error_handler(filename, line, function, err, fmt):
    pass

def silence_alsa():
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    global c_error_handler
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    try:
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
    except Exception as e:
        pass

silence_alsa()

# --- INITIALISATION ---
UDP_IP = "0.0.0.0"
PORT_VIDEO, PORT_AUDIO = 5005, 5006
state_manager = MultimodalState()

# --- CONFIGURATION RETOUR ROBOT ---
PEPPER_IP = "192.168.1.101" # À adapter avec l'IP réelle du robot
PORT_RETOUR = 5007
sock_retour = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)



# --- THREAD AUDIO (SYSTÈME DOUBLE BUFFER) ---
def run_audio_listening():
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True, frames_per_buffer=1024)
    
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_audio.settimeout(1.0) # Permet de vérifier la variable 'running' régulièrement
    sock_audio.bind((UDP_IP, PORT_AUDIO))
    
    buffer_ton = []
    buffer_texte = []
    
    print("Écoute audio activée...")
    while True:
        try:
            data, _ = sock_audio.recvfrom(8192)
            raw = np.frombuffer(data, dtype=np.int16)
            raw_reshaped = raw.reshape(-1, 4)
            audio_mono = raw_reshaped.mean(axis=1).astype(np.int16)
            
            # DÉSACTIVÉ : Évite le larsen mortel avec le Mixage Stéréo
            audio_stream.write(audio_mono.tobytes()) 
            
            buffer_ton.append(audio_mono)
            buffer_texte.append(audio_mono)
            
            # 1. ANALYSE DU TON (~ 1 seconde)
            if len(buffer_ton) >= 45: 
                segment_ton = np.concatenate(buffer_ton).astype(np.float32) / 32768.0
                threading.Thread(target=audio_analysis_ton_task, args=(segment_ton,), daemon=True).start()
                buffer_ton = []
                
            # 2. ANALYSE DU TEXTE (~ 4.2 secondes)
            if len(buffer_texte) >= 150: 
                segment_texte = np.concatenate(buffer_texte).astype(np.float32) / 32768.0
                threading.Thread(target=audio_analysis_texte_task, args=(segment_texte,), daemon=True).start()
                buffer_texte = []
                
        except socket.timeout:
            continue
        except Exception as e:
            pass
            
    # Nettoyage à la fermeture
    audio_stream.stop_stream()
    audio_stream.close()
    pa.terminate()
    print("Thread audio fermé proprement.")

# --- TÂCHES ASYNCHRONES ---
def audio_analysis_ton_task(segment):
    if not lock_ton.acquire(blocking=False):
        return
    try:
        rms = np.sqrt(np.mean(segment**2))
        if rms > 0.01:
            res_aud = get_acoustic_vad(segment, sampling_rate=48000)
            if res_aud:
                v = (res_aud['valence'] * 2) - 1
                a = (res_aud['arousal'] * 2) - 1
                d = (res_aud['dominance'] * 2) - 1
                state_manager.update("audio", [v, a, d])
                print(f"[TON] Valence: {res_aud['valence']:.2f} Arousal: {res_aud['arousal']:.2f} Dominance: {res_aud['dominance']:.2f}")
    finally:
        lock_ton.release()

def audio_analysis_texte_task(segment):
    if not lock_texte.acquire(blocking=False):
        return
    try:
        rms = np.sqrt(np.mean(segment**2))
        if rms > 0.01:
            texte, _ = get_text_vad(segment)
            if texte:
                print(f"[STT] {texte}")
    finally:
        lock_texte.release()

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
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                
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