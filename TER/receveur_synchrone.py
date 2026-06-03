import socket
import cv2
import numpy as np
import threading
import pyaudio
import time
import os
import wave
import csv
from datetime import datetime
from ctypes import *
import json
import matplotlib
import re
import random
matplotlib.use('Agg') # Force Matplotlib à ne pas ouvrir de fenêtre
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, iirnotch
from scipy.signal import find_peaks
import noisereduce as nr
import http.server
import socketserver

from vidToVAD import get_visual_vad
from audToVAD import get_acoustic_vad, get_text_vad
from MultimodalState import MultimodalState 

THRESHOLD_SILENCE =500.0   # Seuil de volume pour détecter la voix
IPU_CHUNKS_LIMIT = 10       # ~300ms de silence requis pour valider une pause
MIN_PHRASE_CHUNKS = 10      # ~400ms d'audio minimum requis pour justifier un STT (évite les raclements de gorge)
MAX_PHRASE_CHUNKS = 150

# FILTRE HIGH PASS
CUTOFF_FREQ = 150.0  
SAMPLE_RATE = 16000.0
nyq = 0.5 * SAMPLE_RATE
normal_cutoff = CUTOFF_FREQ / nyq
b, a = butter(4, normal_cutoff, btype='high', analog=False)

# 1. Filtre Passe-Haut (High-Pass) pour le grondement sourd
# On monte à 300Hz pour supprimer le gros bloc de ta liste
def create_hp(cutoff=300.0, fs=16000.0):
    nyq = 0.5 * fs
    low = cutoff / nyq
    return butter(4, low, btype='high')

b_hp, a_hp = create_hp()

# 2. Filtre Notch pour le sifflement aigu majeur (auton de 3000Hz)
# On prend une bande un peu plus large (Q=20 au lieu de 30) pour ratisser large
def create_notch(freq=3000.0, fs=16000.0, q=20.0):
    nyq = 0.5 * fs
    w0 = freq / nyq
    return iirnotch(w0, q)

b_n, a_n = create_notch(2916.5) # On cible le pic le plus probable

def apply_high_pass(data_int16):
    """ Filtre léger pour enlever l'infra-basse avant la réduction de bruit """
    x = data_int16.astype(np.float32)
    # On utilise b_hp, a_hp que tu as déjà défini (coupure 300Hz)
    x = lfilter(b_hp, a_hp, x)
    return x.astype(np.int16)

# --- CONFIGURATION ENREGISTREMENT ---
CSV_DIR = "csv"
if not os.path.exists(CSV_DIR):
    os.makedirs(CSV_DIR)

# Création du dossier images s'il n'existe pas
if not os.path.exists("images"):
    os.makedirs("images")

recording = False
csv_file = None
csv_writer = None

# Variables pour le suivi des images
images_list = []
current_image_idx = 0

noise_profile = None

# --- VERROUS DE SÉCURITÉ ---
lock_ton = threading.Lock()
lock_texte = threading.Lock()

# --- VARIABLES PARTICIPANT ---
participant_id = ""
participant_dir = ""

def run_http_server():
    """ Héberge le dossier local sur le port 8000 pour la tablette de Pepper """
    socketserver.TCPServer.allow_reuse_address = True
    Handler = http.server.SimpleHTTPRequestHandler
    httpd = socketserver.TCPServer(("", 8000), Handler)
    print("🌐 Serveur Web local démarré sur le port 8000")
    httpd.serve_forever()

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
PORT_VIDEO, PORT_AUDIO, PORT_MOTION = 5005, 5006, 5009
state_manager = MultimodalState()

PEPPER_NAME = "Pepper.local" # Ou l'IP directe (ex: "192.168.1.50")
# --- FIX SACCADES : On résout l'IP une seule fois au démarrage ---
try:
    PEPPER_IP_RESOLVED = socket.gethostbyname(PEPPER_NAME)
    print(f"IP du robot trouvée : {PEPPER_IP_RESOLVED}")
except Exception as e:
    print(f"Attention, impossible de résoudre {PEPPER_NAME}. On garde l'adresse brute.")
    PEPPER_IP_RESOLVED = PEPPER_NAME
# ------------------------------------------------------------------

PORT_RETOUR = 5007
sock_retour = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

def envoyer_commande_bras(commande):
    """ Envoie l'ordre 7 ou 8 à Pepper """
    try:
        sock_retour.sendto(commande.encode('utf-8'), (PEPPER_IP_RESOLVED, PORT_MOTION))
    except Exception as e:
        print(f"Erreur envoi bras : {e}")

def get_next_participant_number():
    """ Scanne le dossier CSV_DIR pour trouver le prochain numéro disponible """
    if not os.path.exists(CSV_DIR):
        os.makedirs(CSV_DIR)
        return 1
    
    # On cherche les dossiers de type 'Participant_XXX_...'
    folders = [f for f in os.listdir(CSV_DIR) if os.path.isdir(os.path.join(CSV_DIR, f))]
    
    existing_ids = []
    for f in folders:
        # Regex pour extraire le nombre juste après 'Participant_'
        match = re.match(r"Participant_(\d+)", f)
        if match:
            existing_ids.append(int(match.group(1)))
    
    if not existing_ids:
        return 1
    
    return max(existing_ids) + 1

def init_participant_session():
    global participant_id, participant_dir, images_list, current_image_idx
    
    # 1. Génération automatique de l'ID
    next_num = get_next_participant_number()
    p_id_str = f"{next_num:03d}" # Formate en 001, 002, etc.

    print("\n" + "="*50)
    print(f"🎓 SESSION PARTICIPANT n°{p_id_str}")
    print("="*50)
    
    # 2. On ne demande plus que le nécessaire pour les stats
    sexe = input("Sexe (M/F/Autre) : ").upper()
    age = input("Age : ")
    
    participant_id = f"Participant_{p_id_str}_{sexe}_{age}"
    participant_dir = os.path.join(CSV_DIR, participant_id)
    
    if not os.path.exists(participant_dir):
        os.makedirs(participant_dir)
        
    # Sauvegarde des métadonnées pour faciliter vos moyennes plus tard
    meta_path = os.path.join(participant_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump({
            "id_numerique": next_num,
            "sexe": sexe, 
            "age": age, 
            "date": datetime.now().isoformat()
        }, f, indent=4)

    # Réinitialisation de la liste d'images pour ce participant
    images_list = [f for f in os.listdir("images") if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images_list)
    current_image_idx = 0
    
    # NOUVEAU : Sélection d'une image aléatoire pour le toucher
    if len(images_list) > 0:
        index_toucher = random.randint(1, len(images_list))
        print(f"\n✅ Dossier créé : {participant_dir}")
        print(f"🔀 Liste de {len(images_list)} images préparée et mélangée.")
        # Ajout visuel fort
        print("\n" + "!"*50)
        print(f"⚠️ INSTRUCTION : Le robot devra toucher le participant sur l'image n° {index_toucher} ! ⚠️")
        print("!"*50 + "\n")
    
    print("Appuyez sur 'r' dans la fenêtre vidéo pour lancer la première image.")

# --- LOGIQUE D'ENREGISTREMENT ---
def toggle_image_session():
    """ 
    Bascule (On/Off) l'enregistrement. 
    Arrête la session en cours ou démarre la session pour l'image suivante.
    """
    global recording, csv_file, csv_writer, images_list, current_image_idx
    
    # --- CAS 1 : On enregistrait déjà -> On ARRETE ---
    if recording:
        recording = False
        if csv_file:
            csv_file.close()
            
        # On retrouve le nom de l'image qu'on vient de terminer
        image_terminee = images_list[current_image_idx - 1] if current_image_idx > 0 else "inconnue"
        
        print("\n" + "="*50)
        print(f"⏹️ ENREGISTREMENT ARRÊTÉ pour : {image_terminee}")
        print("Appuyez de nouveau sur 'r' pour passer à la photo suivante.")
        print("="*50 + "\n")
        return

    # --- CAS 2 : On était en pause -> On DÉMARRE la prochaine image ---
    
    # 1. Charger la liste des images si c'est le tout début
    if not images_list:
        print("\n⚠️ ERREUR : La liste d'images est vide. Avez-vous initialisé le participant ?")
        return

    # Vérifier si on a fait toutes les images
    if current_image_idx >= len(images_list):
        print("\n" + "="*50)
        print("🎉 TOUTES LES IMAGES ONT ÉTÉ DÉCRITES POUR CE PARTICIPANT !")
        print("Fermeture automatique du script.") # Petit changement de texte
        print("="*50 + "\n")
        return "QUITTER"
        
    # Récupérer l'image suivante
    image_name = images_list[current_image_idx]
    current_image_idx += 1
    
    print("\n" + "*"*50)
    print(f"👉 NOUVELLE IMAGE AFFICHÉE : {image_name} ({current_image_idx}/{len(images_list)})")
    print("*"*50)
    
    # --- MODIFICATION ICI : On sauvegarde directement dans le dossier du participant ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # On inclut le nom de l'image dans le nom du fichier CSV
    filename = os.path.join(participant_dir, f"{image_name}_{timestamp}.csv")
    
    try:
        csv_file = open(filename, mode='w', newline='')
        csv_writer = csv.writer(csv_file)
        header = ["timestamp", "v", "a", "vv", "av", "va", "aa", "vt", "at", "mark"]
        csv_writer.writerow(header)
        recording = True
        
        log_to_csv(mark_value=0)
        
        print(f"--- ⏺ ENREGISTREMENT DÉMARRÉ : {filename} ---")
        print("🎙️ Le système enregistre maintenant les réactions !")
    except Exception as e:
        print(f"Erreur lors de la création du CSV de session : {e}")

def log_to_csv(mark_value=0):
    global recording, csv_writer
    if recording and csv_writer:
        try:
            now = time.time()
            # On récupère toutes les variables depuis le state_manager
            vf, af = state_manager.current_fusion
            vv, av = state_manager.data["vision"]
            va, aa = state_manager.data["audio"]
            vt, at = state_manager.data["texte"]
            
            # Ligne à écrire (Format: timestamp, v, a, vv, av, va, aa, vt, at, mark)
            row = [now, vf, af, vv, av, va, aa, vt, at, mark_value]
            csv_writer.writerow(row)
        except Exception as e:
            print(f"Erreur d'écriture CSV: {e}")
            
def log_stt_text(texte, valence, arousal):
    """ Sauvegarde le texte reconnu et ses scores dans le dossier du participant """
    global participant_dir
    
    # Sécurité : on s'assure que le dossier participant est bien créé
    if not participant_dir or not os.path.exists(participant_dir):
        return 

    stt_file_path = os.path.join(participant_dir, "stt_transcriptions.csv")
    file_exists = os.path.exists(stt_file_path)

    try:
        # On utilise encoding='utf-8' impérativement pour les accents français (é, à, ç)
        with open(stt_file_path, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # Si c'est la première phrase du participant, on écrit l'en-tête
            if not file_exists:
                writer.writerow(["Timestamp", "Texte", "Valence", "Arousal"])
            
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # On écrit la ligne
            writer.writerow([now, texte, round(valence, 3), round(arousal, 3)])
            
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du texte STT: {e}")

def log_historical_to_csv(t_timestamp, fusion, vision, audio, texte, mark_value=0):
    """ Écrit une ligne CSV pour le découpage rétroactif """
    global recording, csv_writer
    if recording and csv_writer:
        try:
            row = [t_timestamp, fusion[0], fusion[1], 
                   vision[0], vision[1], audio[0], audio[1], 
                   texte[0], texte[1], mark_value]
            csv_writer.writerow(row)
        except Exception as e:
            print(f"Erreur d'écriture CSV Historique: {e}")

def analyser_bruit_pepper(buffer_audio, rate=16000):
    """ Génère un graphique comparatif Avant/Après filtrage """
    try:
        print("\n[DIAGNOSTIC] Analyse spectrale (Avant/Après) en cours...")
        
        # 1. Traitement du signal BRUT
        segment_brut = np.concatenate(buffer_audio).astype(np.float32) / 32768.0
        n = len(segment_brut)
        freq = np.fft.rfftfreq(n, d=1./rate)
        spectre_brut = np.abs(np.fft.rfft(segment_brut))
        
        # 2. Traitement du signal FILTRÉ
        buffer_filtre = [apply_high_pass(chunk) for chunk in buffer_audio]
        segment_filtre = np.concatenate(buffer_filtre).astype(np.float32) / 32768.0
        spectre_filtre = np.abs(np.fft.rfft(segment_filtre))

        # 3. Création du graphique superposé
        plt.figure(figsize=(14, 7))
        plt.plot(freq, spectre_brut, label='Brut (Moteurs + Ventilos)', color='red', alpha=0.5)
        plt.plot(freq, spectre_filtre, label='Filtré (High-Pass 150Hz)', color='blue', alpha=0.8)
        
        plt.yscale('log')
        plt.title("Comparaison Spectrale de Pepper : Avant / Après Filtrage (16 kHz)")
        plt.xlabel("Fréquence (Hz)")
        plt.ylabel("Intensité")
        
        # À 16kHz, la fréquence de Nyquist est 8000Hz. C'est parfait pour voir toute la bande !
        plt.xlim(0, 8000) 
        plt.grid(True, which="both", alpha=0.3)
        plt.axvline(x=150, color='black', linestyle='--', label='Coupure (150Hz)')
        plt.legend()
        
        filename = "diagnostic_comparatif.png"
        plt.savefig(filename)
        plt.close()
        print(f"[DIAGNOSTIC] Fichier généré : {filename} (Ouvre-le pour voir la différence !)")
    except Exception as e:
        print(f"[ERREUR DIAGNOSTIC] {e}")

def save_wav(filename, frames, rate):
    """Fonction utilitaire pour écrire le fichier sur le disque"""
    wf = wave.open(filename, 'wb')
    wf.setnchannels(1)
    wf.setsampwidth(2) # 2 octets pour du int16
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()
    
def detecter_sifflements(buffer_audio, rate=16000):
    # 1. Calcul du spectre moyen
    segment = np.concatenate(buffer_audio).astype(np.float32) / 32768.0
    n = len(segment)
    freq = np.fft.rfftfreq(n, d=1./rate)
    spectre = np.abs(np.fft.rfft(segment))
    
    # 2. Trouver les pics qui dépassent de la moyenne (bruit de ventilo)
    # On cherche des pics très fins (prominence)
    peaks, properties = find_peaks(spectre, prominence=max(spectre)*0.05)
    
    frequences_parasites = freq[peaks]
    
    print("\n--- DIAGNOSTIC SIFFLEMENTS ---")
    if len(frequences_parasites) > 0:
        print(f"Fréquences détectées à filtrer (Hz) :")
        for f in frequences_parasites:
            if f > 500: # On ignore les basses déjà gérées
                print(f" > {f:.1f} Hz")
    else:
        print("Aucun sifflement aigu majeur détecté par l'algorithme.")
    
    return frequences_parasites

def apply_smart_denoise(data_int16):
    global noise_profile
    
    # 1. On applique TOUJOURS le filtre High-Pass en premier (pour virer l'infra-basse)
    hp_audio = apply_high_pass(data_int16)
    
    if noise_profile is None:
        # Pendant le calibrage, on renvoie juste le son sans basses
        return hp_audio
    
    # 2. Conversion en float NORMALISÉ (-1.0 à 1.0) pour correspondre au profil
    x_float = hp_audio.astype(np.float32) / 32768.0
    
    # 3. Réduction de bruit (cette fois, les échelles correspondent !)
    reduced_float = nr.reduce_noise(
        y=x_float, 
        sr=int(SAMPLE_RATE), 
        y_noise=noise_profile, 
        stationary=True, 
        prop_decrease=1.0
    )
    
    # 4. Reconversion en int16 pour tes haut-parleurs et le calcul d'énergie
    reduced_int = (reduced_float * 32768.0).astype(np.int16)
    
    return reduced_int

def enregistrer_wav_diagnostic(buffer_audio, rate=16000):
    global noise_profile
    try:
        buffer_pre_filtered = [apply_high_pass(chunk) for chunk in buffer_audio]
        noise_profile = np.concatenate(buffer_pre_filtered).astype(np.float32) / 32768.0
        
        # --- SAUVEGARDE DU PROFIL ---
        np.save(NOISE_PROFILE_FILE, noise_profile)
        print(f"💾 Profil de bruit sauvegardé dans {NOISE_PROFILE_FILE}")
        filename = "bruit_pepper_brut.wav"
        print(f"\n[DIAGNOSTIC] Génération du fichier audio : {filename}...")
        
        # Conversion du buffer (liste de arrays int16) en un seul bloc binaire
        data_all = b''.join([chunk.tobytes() for chunk in buffer_audio])
        
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)          # Mono
            wf.setsampwidth(2)          # 16 bits = 2 octets
            wf.setframerate(rate)       # 16000 Hz
            wf.writeframes(data_all)
            
        print(f"[DIAGNOSTIC] Fichier WAV prêt ! Tu peux l'ouvrir dans Audacity.")
    except Exception as e:
        print(f"[ERREUR WAV] {e}")

# Nom du fichier de profil
NOISE_PROFILE_FILE = "noise_profile.npy"

def charger_profil_bruit():
    global noise_profile
    if os.path.exists(NOISE_PROFILE_FILE):
        try:
            noise_profile = np.load(NOISE_PROFILE_FILE)
            print(f"✅ Profil de bruit chargé avec succès depuis {NOISE_PROFILE_FILE}")
        except Exception as e:
            print(f"⚠️ Erreur lors du chargement du profil: {e}")
            noise_profile = None
    else:
        print("ℹ️ Aucun profil de bruit trouvé. Calibrage nécessaire.")

def run_audio_listening():
    pa = pyaudio.PyAudio()
    # Mise à jour du flux de sortie audio pour écouter en 16kHz
    audio_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=int(SAMPLE_RATE), output=True)
    sock_audio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_audio.bind((UDP_IP, PORT_AUDIO))
    
    current_phrase_buffer = []
    diag_buffer = [] 
    silence_counter = 0
    start_phrase_time = 0

    print(f"\n--- ÉCOUTE AUDIO ACTIVÉE ({int(SAMPLE_RATE)}Hz) ---")
    
    # --- MODIFICATION 1 : Message d'attente uniquement si nécessaire ---
    if noise_profile is None:
        print("🛑 ATTENTION : RESTEZ TOTALEMENT SILENCIEUX (environ 6 secondes).")
        print("Le système capture le bruit des ventilateurs de Pepper...")
    else:
        print("✅ Profil de bruit détecté ! Le système est prêt à écouter immédiatement.")
    
    while True:
        try:
            data, _ = sock_audio.recvfrom(65535)
            raw_audio = np.frombuffer(data, dtype=np.int16)
            
            # --- MODIFICATION 2 : On ne capture le diagnostic QUE si on n'a pas de profil ---
            if noise_profile is None:
                if len(diag_buffer) < 100: 
                    diag_buffer.append(raw_audio)
                    if len(diag_buffer) == 100:
                        threading.Thread(target=enregistrer_wav_diagnostic, args=(diag_buffer, int(SAMPLE_RATE))).start()
                # On ne traite pas la voix tant que le profil n'est pas fait
                continue
            
            # ==============================================================
            # TRAITEMENT DE LA VOIX (Exécuté instantanément si le profil existe)
            # ==============================================================
            
            # Application du filtre (le filtre est mathématiquement calé sur 16kHz)
            audio_mono = apply_high_pass(raw_audio)
            
            # Retour Audio (Tu entends le son filtré en 16k - à commenter si écho)
            audio_stream.write(audio_mono.tobytes())
            
            # --- AJOUT ICI : ON COUPE L'ÉCOUTE SI ON EST EN PAUSE ---
            if not recording:
                # On vide le buffer pour éviter de garder un bout de mot prononcé juste avant la pause
                current_phrase_buffer = [] 
                silence_counter = 0
                continue # On passe direct au paquet suivant sans analyser l'énergie
            # --------------------------------------------------------
            
            energy = np.abs(audio_mono).mean()
            
            # On écoute la voix
            if energy > THRESHOLD_SILENCE:
                if len(current_phrase_buffer) == 0:
                    start_phrase_time = time.time()
                    print("\n[ÉCOUTE] ", end="", flush=True)
                
                current_phrase_buffer.append(audio_mono)
                silence_counter = 0
                print(".", end="", flush=True)

                if len(current_phrase_buffer) % 15 == 0:
                    segment_ton = np.concatenate(current_phrase_buffer).astype(np.float32) / 32768.0
                    threading.Thread(target=audio_analysis_ton_task, args=(segment_ton,), daemon=True).start()
            else:
                if len(current_phrase_buffer) > 0:
                    if silence_counter == 0:
                        print(" [PAUSE] ", end="", flush=True)
                    
                    silence_counter += 1
                    print("-", end="", flush=True)
                    
                    if silence_counter >= IPU_CHUNKS_LIMIT:
                            if len(current_phrase_buffer) >= MIN_PHRASE_CHUNKS:
                                print(f"\n[IPU] Phrase validée ({len(current_phrase_buffer)} chunks). Envoi STT...")
                                
                                                            # 1. On regroupe la phrase captée (qui a juste le filtre léger)
                                segment_brut = np.concatenate(current_phrase_buffer)
                                
                                # 2. On applique la vraie suppression de bruit complète (profil .npy)
                                segment_propre_int16 = apply_smart_denoise(segment_brut)
                                
                                # 3. Conversion en float pour Whisper
                                segment_propre = segment_propre_int16.astype(np.float32) / 32768.0
                                
                                end_phrase_time = time.time()
                                threading.Thread(target=audio_analysis_texte_task, args=(segment_propre, start_phrase_time, end_phrase_time), daemon=True).start()
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
            res_aud = get_acoustic_vad(segment, sampling_rate=16000)
            if res_aud:
                print(f"\n {time.time()} - Ton acoustique : V={res_aud['valence']:.2f}, A={res_aud['arousal']:.2f}")
                state_manager.update("audio", [(res_aud['valence']*2)-1, (res_aud['arousal']*2)-1])
    finally: lock_ton.release()

def audio_analysis_texte_task(segment, start_time, end_time):
    """ Analyse STT + VA Textuel déclenchée par la fin d'une phrase """
    try:
        # Appel à ton moteur STT (ex: Whisper + DistilBERT)
        texte, va_scores = get_text_vad(segment) 
        
        if texte and va_scores:
            # 1. Mise à jour de la modalité texte (VA et texte)
            print(f"\n[STT] Texte reconnu : {texte}")
            print(f"\n on envoie à state_manager text : {va_scores[:2]}")
            
            # --- AJOUT ICI : Sauvegarde dans le fichier du participant ---
            log_stt_text(texte, va_scores[0], va_scores[1])
            # -----------------------------------------------------------
            
            state_manager.update("texte", va_scores[:2])

            print(f"\n--- DÉCOUPAGE RÉTROACTIF (Tranches de 2s) ---")
            current_start = start_time
            fusion_finale = np.array([0.0, 0.0])
            
            while current_start < end_time:
                current_end = min(current_start + 2.0, end_time)
                fusion, v_v, v_a, v_t = state_manager.get_detailed_state_window(current_start, current_end)
                log_historical_to_csv(current_end, fusion, v_v, v_a, v_t, mark_value=0)
                duree_tranche = current_end - current_start
                print(f" -> Tranche {duree_tranche:.1f}s | FUS: {fusion[0]:+.2f}, {fusion[1]:+.2f} | VIS: {v_v[0]:+.2f}, {v_v[1]:+.2f} | TON: {v_a[0]:+.2f}, {v_a[1]:+.2f}")
                current_start += 2.0
                fusion_finale = fusion # On garde la dernière pour l'envoi au robot
            
            print("---------------------------------------------\n")

            state_manager.current_fusion = fusion_finale
            envoyer_debug_robot( True, mouvement=False)
            
    except Exception as e:
        print("\nErreur STT Task:", e)

def envoyer_debug_robot(visage_detecte, mouvement=False):
    """ Envoie les données au robot Pepper """
    try:
        v_now, a_now = state_manager.current_fusion
        
        # --- NOUVEAU : Récupérer le nom de l'image en cours ---
        image_actuelle = ""
        if recording and len(images_list) > 0 and current_image_idx > 0:
            image_actuelle = images_list[current_image_idx - 1]

        data = {
            "status": "ok" if visage_detecte else "none",
            "move": mouvement,
            "v": round(float(v_now), 2),
            "a": round(float(a_now), 2),
            "image": image_actuelle # On ajoute l'image au paquet
        }

        print(f"📡 [RESEAU] Envoi vers Pepper -> Image: '{image_actuelle}' | V: {data['v']} | A: {data['a']}")

        sock_retour.sendto(json.dumps(data).encode('utf-8'), (PEPPER_IP_RESOLVED, PORT_RETOUR))
    except Exception as e:
        pass



# --- BOUCLE PRINCIPALE ---
def main():
    print("--- SYSTEME MULTIMODAL VA PRÊT (AVEC LOGS) ---")
    
    # 1. On règle l'administratif d'abord (le code attend tes inputs ici)
    init_participant_session()

    # 2. ENSUITE on lance les processus en arrière-plan
    charger_profil_bruit()
    threading.Thread(target=run_audio_listening, daemon=True).start()
    threading.Thread(target=run_http_server, daemon=True).start()
    
    # 3. Initialisation de la vidéo
    sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_video.bind((UDP_IP, PORT_VIDEO))
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    last_robot_update = 0
    last_face_log = 0
    nb_frame_logs = 0
    bras_en_flexion = False

    try:
        while True:
            data_v, _ = sock_video.recvfrom(65535)
            nparr = np.frombuffer(data_v, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is not None:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # Detection visage
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
                visage_detecte = len(faces) > 0

                if visage_detecte: #
                    # Mise à jour Vision
                    va_v = get_visual_vad(frame)

                    nb_frame_logs += 1

                    if nb_frame_logs >= 200 :
                        print(f"\n on envoie à state_manager vision : {va_v[:2]}")
                        nb_frame_logs = 0

                
                    state_manager.update("vision", va_v[:2])
                    if time.time() - last_face_log > 1.0: # Log toutes les secondes
                        
                        last_face_log = time.time()
                
                # Envoi tablette (toutes les 500ms)
                if time.time() - last_robot_update > 0.5:
                    # FIX : on unpacke seulement 2 valeurs maintenant
                    last_robot_update = time.time()
                cv2.imshow("Analyse VA", frame)
                
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'): 
                break
            elif key == ord('r'):
                signal = toggle_image_session() # <--- MODIFICATION ICI
                if signal == "QUITTER":         # <--- AJOUT
                    break                       # <--- AJOUT : On casse la boucle infinie
                envoyer_debug_robot(True, mouvement=False)
            elif key == ord('m'): # Pour ton bouton mouvement/mark
                print("M APPUYE")
                log_to_csv(mark_value=1)
                envoyer_debug_robot(True, mouvement=True)
            elif key == ord('7'):
                print("💪 Commande : Flexion du coude")
                envoyer_commande_bras("7")
                bras_en_flexion = True # On mémorise qu'on s'est préparé à toucher
                
            elif key == ord('8'):
                print("🦾 Commande : Extension du coude")
                envoyer_commande_bras("8")
                
                # Si on fait une extension ALORS qu'on était en flexion -> TOUCHER !
                if bras_en_flexion:
                    print("\n" + "🎯"*10)
                    print("🎯 TOUCHER DÉTECTÉ ET ENREGISTRÉ (MARK = 1) !")
                    print("🎯"*10 + "\n")
                    
                    log_to_csv(mark_value=1) # On marque le CSV automatiquement
                    envoyer_debug_robot(visage_detecte, mouvement=True)
                    bras_en_flexion = False # On réinitialise pour le prochain
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()