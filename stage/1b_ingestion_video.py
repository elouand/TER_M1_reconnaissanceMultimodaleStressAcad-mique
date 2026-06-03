# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import scipy.io.wavfile as wav
import librosa
import math
import subprocess

# ==========================================
# CONFIGURATION
# ==========================================
VIDEO_PATH = "sujet_recola_test.mp4"  # Chemin vers ta vidéo dataset
VERSION = "v1.0"                      # Version actuelle de l'architecture
WINDOW_SIZE = 2.0                     # Taille de la fenêtre temporelle en secondes
TEMP_WAV = "audio_temp_extraction.wav" # Fichier temporaire pour contourner le bug librosa

# ==========================================
# PRÉPARATION DES DOSSIERS
# ==========================================
nom_video = os.path.splitext(os.path.basename(VIDEO_PATH))[0]
DATA_DIR = os.path.join("donnees_brutes", VERSION, nom_video)

# Création du dossier d'extraction
os.makedirs(DATA_DIR, exist_ok=True)

def extraire_donnees():
    print(f"--- 🎬 DÉBUT DE L'INGESTION RAW (Version : {VERSION}) ---")
    print(f"Vidéo cible : {VIDEO_PATH}")
    
    if not os.path.exists(VIDEO_PATH):
        print(f"❌ Erreur : La vidéo '{VIDEO_PATH}' est introuvable.")
        return

    # 1. EXTRACTION AUDIO ROBUSTE (via FFmpeg)
    print("\n⚡ Extraction instantanée de l'audio via FFmpeg...")
    try:
        # On force FFmpeg à extraire la piste audio en WAV 16-bit (sans perte)
        # stderr=subprocess.DEVNULL cache les logs verbeux de FFmpeg dans le terminal
        subprocess.run([
            "ffmpeg", "-y", "-i", VIDEO_PATH, 
            "-vn", "-acodec", "pcm_s16le", TEMP_WAV
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("❌ Erreur : FFmpeg a échoué. Est-il bien installé sur le PC (sudo apt install ffmpeg) ?")
        return

    # 2. CHARGEMENT AUDIO DEPUIS LE WAV (Instantané et sans bug)
    print("🎙️ Chargement de l'audio (Qualité native d'origine)...")
    audio_data, original_sr = librosa.load(TEMP_WAV, sr=None, mono=True)
    duree_totale_audio = len(audio_data) / original_sr
    
    # Nettoyage du fichier temporaire
    if os.path.exists(TEMP_WAV):
        os.remove(TEMP_WAV)

    # 3. CHARGEMENT VIDÉO
    print("🎥 Ouverture du flux vidéo...")
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps_source = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    duree_totale = math.floor(duree_totale_audio)
    nb_windows = int(duree_totale // WINDOW_SIZE)
    
    print(f"📊 Résolution source : {width}x{height} à {fps_source:.2f} FPS")
    print(f"📊 Fréquence audio native : {original_sr} Hz")
    print(f"📊 Durée totale : {duree_totale}s | Nombre de tranches ({WINDOW_SIZE}s) : {nb_windows}\n")

    # 4. DÉCOUPAGE TEMPOREL (Fenêtrage 100% sans perte)
    for w in range(nb_windows):
        t_start = w * WINDOW_SIZE
        t_end = t_start + WINDOW_SIZE
        
        # --- AUDIO ---
        start_sample = int(t_start * original_sr)
        end_sample = int(t_end * original_sr)
        audio_chunk = audio_data[start_sample:end_sample]
        
        # Conversion float32 -> int16 (format standard lisible partout)
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        
        chemin_audio = os.path.join(DATA_DIR, f"audio_{t_start:.1f}_{t_end:.1f}.wav")
        wav.write(chemin_audio, original_sr, audio_int16)

        # --- VIDÉO ---
        dossier_frames = os.path.join(DATA_DIR, f"frames_{t_start:.1f}_{t_end:.1f}")
        os.makedirs(dossier_frames, exist_ok=True)
        
        # Calcul du nombre de frames exact dans cette fenêtre
        frames_a_lire = int(WINDOW_SIZE * fps_source)
        saved_frames = 0
        
        for _ in range(frames_a_lire):
            ret, frame = cap.read()
            if not ret:
                break
            
            # Sauvegarde de la frame dans sa résolution d'origine
            nom_frame = os.path.join(dossier_frames, f"frame_{saved_frames:04d}.jpg")
            cv2.imwrite(nom_frame, frame)
            saved_frames += 1
            
        print(f"✅ Tranche [{t_start:.1f}s - {t_end:.1f}s] extraite : {saved_frames} images, audio natif.")

    cap.release()
    print(f"\n🎉 Ingestion RAW terminée avec succès !")
    print(f"📁 Toutes les données brutes ont été stockées dans : {DATA_DIR}")

if __name__ == "__main__":
    extraire_donnees()