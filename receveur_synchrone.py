import socket
import cv2
import numpy as np
import threading
import pyaudio
import time
import os
import csv
from datetime import datetime
from ctypes import *
import json
import matplotlib
matplotlib.use('Agg') # Force Matplotlib à ne pas ouvrir de fenêtre
import matplotlib.pyplot as plt
import keyboard

from vidToVAD import get_visual_vad
from audToVAD import get_acoustic_vad, get_text_vad
from MultimodalState import MultimodalState 

THRESHOLD_SILENCE =500.0   # Seuil de volume pour détecter la voix
IPU_CHUNKS_LIMIT = 10       # ~300ms de silence requis pour valider une pause
MIN_PHRASE_CHUNKS = 10      # ~400ms d'audio minimum requis pour justifier un STT (évite les raclements de gorge)
MAX_PHRASE_CHUNKS = 150

# --- CONFIGURATION ENREGISTREMENT ---
CSV_DIR = "csv"
if not os.path.exists(CSV_DIR):
    os.makedirs(CSV_DIR)

recording = False
csv_file = None
csv_writer = None

# --- VERROUS DE SÉCURITÉ ---
lock_ton = threading.Lock()
lock_texte = threading.Lock()

# (Fonctions silence_alsa et py_error_handler inchangées...)
def py_error_handler(filename, line, function, err, fmt): pass
def silence_alsa():
    ERROR_HANDLER_FUNC = CFUNCTYPE(None, c_char_p, c_int, c_char_p, c_int, c_char_p)
    global c_error_handler
    c_error_handler = ERROR_HANDLER_FUNC(py_error_handler)
    try:
        asound = cdll.LoadLibrary('libasound.so.2')
        asound.snd_lib_error_set_handler(c_error_handler)
    except Exception: pass
silence_alsa()

# --- INITIALISATION ---
UDP_IP = "0.0.0.0"
PORT_VIDEO, PORT_AUDIO = 5005, 5006
state_manager = MultimodalState()

PEPPER_NAME = "Pepper.local" 
PORT_RETOUR = 5007
sock_retour = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# --- LOGIQUE D'ENREGISTREMENT ---
def toggle_recording():
    global recording, csv_file, csv_writer
    if not recording:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(CSV_DIR, f"test_{timestamp}.csv")
        csv_file = open(filename, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        # Header correspondant à ta demande
        header = ["timestamp", "v", "a", "d", "vv", "av", "dv", "va", "aa", "da", "vt", "at", "dt"]
        csv_writer.writerow(header)
        recording = True
        print(f"--- ⏺ ENREGISTREMENT DÉMARRÉ : {filename} ---")
    else:
        recording = False
        if csv_file:
            csv_file.close()
        print("--- ⏹ ENREGISTREMENT ARRÊTÉ ET SAUVEGARDÉ ---")

def analyser_bruit_pepper(buffer_audio, rate=48000):
    """ Génère et sauvegarde le spectre fréquentiel sans bloquer le programme """
    try:
        print("\n[DIAGNOSTIC] Analyse spectrale en cours (Sauvegarde en image)...")
        segment = np.concatenate(buffer_audio).astype(np.float32) / 32768.0
        n = len(segment)
        freq = np.fft.rfftfreq(n, d=1./rate)
        spectre = np.abs(np.fft.rfft(segment))

        plt.figure(figsize=(12, 6))
        plt.semilogy(freq, spectre) 
        plt.title("Spectre Sonore de Pepper (Diagnostic Bruit)")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Intensité")
        plt.xlim(0, 8000) # Concentrons-nous sur la zone de la voix
        plt.grid(True, which="both", alpha=0.3)
        plt.axvline(x=150, color='r', linestyle='--', label='Coupure Basse conseillée')
        plt.legend()
        
        # Sauvegarde le fichier dans ton dossier actuel
        filename = "diagnostic_bruit.png"
        plt.savefig(filename)
        plt.close()
        print(f"[DIAGNOSTIC] Analyse terminée. Fichier généré : {filename}")
    except Exception as e:
        print(f"[ERREUR DIAGNOSTIC] {e}")

def run_audio_listening():
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True)
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_audio.bind((UDP_IP, PORT_AUDIO))
    
    current_phrase_buffer = []
    diag_buffer = [] # Buffer pour le diagnostic
    silence_counter = 0
    start_phrase_time = 0

    print(f"\n--- ÉCOUTE AUDIO ACTIVÉE ---")
    print("Astuce : Le programme collectera les 2 premières secondes pour analyse...")
    
    while True:
        try:
            data, _ = sock_audio.recvfrom(65535)
            audio_mono = np.frombuffer(data, dtype=np.int16)
            audio_stream.write(audio_mono.tobytes())
            
            # --- BLOC DIAGNOSTIC ---
            # On remplit un buffer de diagnostic tant qu'il n'y a pas de parole
            if len(diag_buffer) < 100: # Environ 2 secondes de son
                diag_buffer.append(audio_mono)
                if len(diag_buffer) == 100:
                    # On lance l'analyse dans un thread séparé pour ne pas bloquer l'écoute
                    threading.Thread(target=analyser_bruit_pepper, args=(diag_buffer,)).start()
            # -----------------------

            energy = np.abs(audio_mono).mean()
            
            if energy > THRESHOLD_SILENCE:
                if len(current_phrase_buffer) == 0:
                    start_phrase_time = time.time()
                    print("\n[ÉCOUTE] ", end="", flush=True)
                
                current_phrase_buffer.append(audio_mono)
                silence_counter = 0
                print(".", end="", flush=True)
            else:
                if len(current_phrase_buffer) > 0:
                    if silence_counter == 0:
                        print(" [PAUSE] ", end="", flush=True)
                    
                    silence_counter += 1
                    print("-", end="", flush=True)
                    
                    if silence_counter >= IPU_CHUNKS_LIMIT:
                        if len(current_phrase_buffer) >= MIN_PHRASE_CHUNKS:
                            print(f"\n[IPU] Phrase validée ({len(current_phrase_buffer)} chunks). Envoi STT...")
                            segment = np.concatenate(current_phrase_buffer).astype(np.float32) / 32768.0
                            # Lancement du thread d'analyse texte avec le timestamp original
                            threading.Thread(target=audio_analysis_texte_task, args=(segment, start_phrase_time), daemon=True).start()
                        else:
                            print(f"\n[IPU] Ignoré (Bruit trop court).")
                        
                        current_phrase_buffer = []
                        silence_counter = 0
        except: continue

def audio_analysis_ton_task(segment):
    if not lock_ton.acquire(blocking=False): return
    try:
        rms = np.sqrt(np.mean(segment**2))
        if rms > 0.01:
            res_aud = get_acoustic_vad(segment, sampling_rate=48000)
            if res_aud:
                state_manager.update("audio", [(res_aud['valence']*2)-1, (res_aud['arousal']*2)-1, (res_aud['dominance']*2)-1])
    finally: lock_ton.release()

def audio_analysis_texte_task(segment, start_time):
    """ Analyse STT + VA Textuel déclenchée par la fin d'une phrase """
    try:
        # Appel à ton moteur STT (ex: Whisper + DistilBERT)
        texte, va_scores = get_text_vad(segment) 
        
        if texte and va_scores:
            # 1. Mise à jour de la modalité texte (VA uniquement)
            state_manager.update("texte", va_scores[:2])
            
            # 2. Calcul de la FUSION SYNCHRONISÉE sur le début de la phrase
            # On va chercher dans les buffers la vision et le ton à 'start_time'
            va_fusion = state_manager.get_synced_fusion(start_time)
            
            print("\n" + "="*30)
            print("[STT] Phrase: %s" % texte)
            print("[FUSION SYNC] V: %.2f | A: %.2f" % (va_fusion[0], va_fusion[1]))
            print("="*30)
            
            #envoi au state_manager 
            state_manager.update("fusion", va_fusion[:2])
            
            # 3. On déclenche le mouvement du robot sur la phrase finie
            envoyer_debug_robot(va_fusion, True, mouvement=True)
            
    except Exception as e:
        print("\nErreur STT Task:", e)

def envoyer_debug_robot(va_scores, face_found, mouvement=False):
    """ Envoie les données VA fusionnées au robot Pepper """
    try:
        target_ip = socket.gethostbyname(PEPPER_NAME)
        data = {
            "status": "ok" if face_found else "none",
            "v": round(float(va_scores[0]), 2),
            "a": round(float(va_scores[1]), 2),
            "move": mouvement
        }
        sock_retour.sendto(json.dumps(data).encode('utf-8'), (target_ip, PORT_RETOUR))
    except Exception as e:
        print("Erreur envoi Pepper:", e)

# --- BOUCLE PRINCIPALE ---
def main():
    threading.Thread(target=run_audio_listening, daemon=True).start()
    sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_video.bind((UDP_IP, PORT_VIDEO))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    last_robot_update = 0

    print("--- SYSTEME MULTIMODAL VA PRÊT (AVEC LOGS) ---")

    try:
        while True:
            data_v, _ = sock_video.recvfrom(65535)
            nparr = np.frombuffer(data_v, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # Mise à jour Vision
                va_v = get_visual_vad(frame)
                state_manager.update("vision", va_v[:2])

                # Detection visage
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                
                # Envoi tablette (toutes les 500ms)
                if time.time() - last_robot_update > 0.5:
                    # FIX : on unpacke seulement 2 valeurs maintenant
                    v_now, a_now = state_manager.get_fusion() 
                    envoyer_debug_robot([v_now, a_now], len(faces) > 0, mouvement=False)
                    last_robot_update = time.time()

                cv2.imshow("Analyse VA", frame)
                
            if cv2.waitKey(1) & 0xFF == ord('q'): break
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()