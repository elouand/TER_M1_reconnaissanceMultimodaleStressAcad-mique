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
import keyboard

from vidToVAD import get_visual_vad
from audToVAD import get_acoustic_vad, get_text_vad
from MultimodalState import MultimodalState 

THRESHOLD_SILENCE = 500.0   # Seuil de volume pour détecter la voix
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

# (Tâches audio_analysis_ton_task et audio_analysis_texte_task inchangées...)
def run_audio_listening():
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=48000, output=True, frames_per_buffer=1024)
    
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_audio.settimeout(1.0)
    sock_audio.bind((UDP_IP, PORT_AUDIO))
    
    buffer_ton = []
    current_phrase_buffer = [] 
    silence_counter = 0
    
    print(f"\n--- SYSTÈME PRÊT ---")
    print(f"Écoute audio activée (Mode IPU Intelligent - Seuil: {THRESHOLD_SILENCE})...")
    
    while True:
        try:
            data, _ = sock_audio.recvfrom(65535)
            raw = np.frombuffer(data, dtype=np.int16)
            
            # SÉCURITÉ CRUCIALE ANTI-HÉLICOPTÈRE
            if len(raw) % 4 != 0:
                continue
                
            raw_reshaped = raw.reshape(-1, 4)
            audio_mono = raw_reshaped.mean(axis=1).astype(np.int16)
            
            # 1. Retour Audio
            audio_stream.write(audio_mono.tobytes()) 
            
            # 2. Calcul de l'énergie
            energy = np.abs(audio_mono).mean()
            
            # --- LOGIQUE IPU INTELLIGENTE ---
            if energy > THRESHOLD_SILENCE:
                if len(current_phrase_buffer) == 0:
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
                    
                    # Déclenchement sur silence
                    if silence_counter >= IPU_CHUNKS_LIMIT:
                        # Vérification de la longueur minimale
                        if len(current_phrase_buffer) >= MIN_PHRASE_CHUNKS:
                            print(f"\n[IPU] Phrase validée ({len(current_phrase_buffer)} chunks). Envoi STT...")
                            segment_texte = np.concatenate(current_phrase_buffer).astype(np.float32) / 32768.0
                            threading.Thread(target=audio_analysis_texte_task, args=(segment_texte,), daemon=True).start()
                        else:
                            print(f"\n[IPU] Ignoré (Bruit trop court : {len(current_phrase_buffer)} chunks).")
                        
                        # Reset après traitement ou annulation
                        current_phrase_buffer = []
                        silence_counter = 0

            # --- COUPE-CIRCUIT (Monologue trop long) ---
            if len(current_phrase_buffer) >= MAX_PHRASE_CHUNKS:
                print(f"\n[IPU] Monologue continu (Envoi forcé)...")
                segment_texte = np.concatenate(current_phrase_buffer).astype(np.float32) / 32768.0
                threading.Thread(target=audio_analysis_texte_task, args=(segment_texte,), daemon=True).start()
                
                current_phrase_buffer = []
                silence_counter = 0

            # --- LOGIQUE TON (Fixe ~ 1 seconde) ---
            buffer_ton.append(audio_mono)
            if len(buffer_ton) >= 45: 
                segment_ton = np.concatenate(buffer_ton).astype(np.float32) / 32768.0
                threading.Thread(target=audio_analysis_ton_task, args=(segment_ton,), daemon=True).start()
                buffer_ton = []
                
        except socket.timeout:
            continue
        except Exception as e:
            pass

def audio_analysis_ton_task(segment):
    if not lock_ton.acquire(blocking=False): return
    try:
        rms = np.sqrt(np.mean(segment**2))
        if rms > 0.01:
            res_aud = get_acoustic_vad(segment, sampling_rate=48000)
            if res_aud:
                state_manager.update("audio", [(res_aud['valence']*2)-1, (res_aud['arousal']*2)-1, (res_aud['dominance']*2)-1])
    finally: lock_ton.release()

def audio_analysis_texte_task(segment):
    if not lock_texte.acquire(blocking=False):
        return
    try:
        # Vérification du niveau sonore avant traitement lourd
        rms = np.sqrt(np.mean(segment**2))
        if rms > 0.01:
            # Appel au STT et au VAD Textuel
            texte, vad_scores = get_text_vad(segment)
            
            if texte and texte.strip():
                print(f"\n[STT] Phrase reconnue : \"{texte}\"")
                
                # Si get_text_vad renvoie déjà des scores (ex: via DistilBERT/RoBERTa)
                if vad_scores:
                    # Sécurité : on gère la taille de la liste renvoyée par DistilBERT
                    if len(vad_scores) == 3:
                        v, a, d = vad_scores
                    elif len(vad_scores) == 2:
                        v, a = vad_scores
                        d = 0.0 # Valeur neutre pour la Dominance manquante
                    else:
                        v, a, d = 0.0, 0.0, 0.0

                    state_manager.update("texte", [v, a, d]) 
                    print(f"[VAD TEXTE] V: {v:.2f} | A: {a:.2f} | D: {d:.2f}")
            else:
                print("\n[STT] Aucune parole intelligible détectée dans le segment.")
    except Exception as e:
        print(f"\n[ERREUR STT] : {e}")
    finally:
        lock_texte.release()

def envoyer_debug_robot(vad_scores, face_found, mouvement=False):
    try:
        target_ip = socket.gethostbyname(PEPPER_NAME)
        if face_found and vad_scores is not None:
            v, a, d = vad_scores
            data = {
                "status": "ok",
                "v": round(float(v), 2),    
                "a": round(float(a), 2),
                "d": round(float(d), 2),
                "move": mouvement # On ajoute la clé move ici
            }
        else:
            data = {"status": "none", "move": mouvement}
            
        sock_retour.sendto(json.dumps(data).encode('utf-8'), (target_ip, PORT_RETOUR))
    except: pass

# --- BOUCLE PRINCIPALE ---
def main():
    global recording, csv_writer
    threading.Thread(target=run_audio_listening, daemon=True).start()
    sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_video.bind((UDP_IP, PORT_VIDEO))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    last_robot_update = 0
    print("--- Système Multimodal Prêt ---")
    print("TOUCHES : [R] Enregistrer | [M] Mouvement | [Q] Quitter")
    mouvement_commande = False

    # ... (imports et init identiques)

    def mouvementBras():
        # Fonction de test pour simuler un mouvement de bras
        print("Simuler mouvement de bras (fonction à implémenter selon ton robot)")
        # Ici tu pourrais envoyer une commande spécifique à Pepper pour faire bouger les bras
        # Par exemple, en utilisant une librairie de contrôle de Pepper ou en envoyant un message spécifique via socket


    try:
        while True:
            data_v, _ = sock_video.recvfrom(65535)
            nparr = np.frombuffer(data_v, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                
                # 1. Mise à jour Vision
                v_v, a_v, d_v = get_visual_vad(frame)
                state_manager.update("vision", [v_v, a_v, d_v])

                # 2. Calcul de la Fusion (VAD final)
                v_f, a_f, d_f = state_manager.get_fusion()
                
                # 3. Extraction des données pour le CSV (Sécurisée)
                # On récupère les valeurs actuelles pour chaque modalité
                vv, av, dv = state_manager.data["vision"][:3]
                va, aa, da = state_manager.data["audio"][:3]
                vt, at, dt = state_manager.data["texte"][:3]

                # 4. Écriture CSV si l'enregistrement est actif
                if recording and csv_writer:
                    # On crée la ligne avec : Time, Final(VAD), Vision(VAD), Audio(VAD), Texte(VAD)
                    row = [time.time(), v_f, a_f, d_f, vv, av, dv, va, aa, da, vt, at, dt]
                    csv_writer.writerow(row)

                # 5. Gestion du Robot
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                visage_detecte = len(faces) > 0

                if time.time() - last_robot_update > 0.2:
                    envoyer_debug_robot([v_f, a_f, d_f], visage_detecte, mouvement_commande)
                    mouvement_commande = False
                    last_robot_update = time.time()

                # 6. Affichage écran PC
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"FUSION V:{v_f:.2f} A:{a_f:.2f}", (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                if recording:
                    cv2.circle(frame, (30, 30), 10, (0, 0, 255), -1) # Point rouge REC
                    cv2.putText(frame, "ENREGISTREMENT...", (50, 40), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Analyse Multimodale", frame)
                
            # Gestion des touches clavier
            key = cv2.waitKey(1) & 0xFF
            if key == ord('m') or key == ord('M'):
                mouvement_commande = True
                print("--> Commande MOUVEMENT envoyée !")
            elif key == ord('q'): 
                break
            elif key == ord('r'): 
                toggle_recording()

    finally:
        # On ferme proprement le fichier si on quitte pendant un enregistrement
        if recording:
            toggle_recording()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()