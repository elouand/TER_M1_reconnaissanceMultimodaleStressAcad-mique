# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import scipy.io.wavfile as wav
import librosa
import math
import subprocess
import glob

# ==========================================
# CONFIGURATION
# ==========================================
VERSION = "v1.2"
WINDOW_SIZE = 2.0
VIDEO_DIR = "videos_test"
TEMP_WAV = "audio_temp_extraction.wav"

os.makedirs(VIDEO_DIR, exist_ok=True)

def extraire_donnees_batch():
    print(f"--- 🎬 DÉBUT DE L'INGESTION RAW BATCH (Version : {VERSION}) ---")
    
    videos = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    if not videos:
        print(f"❌ Aucune vidéo trouvée dans le dossier '{VIDEO_DIR}'.")
        return

    print(f"📁 {len(videos)} vidéos détectées pour l'ingestion.")

    for video_path in videos:
        nom_video = os.path.splitext(os.path.basename(video_path))[0]
        data_dir = os.path.join("donnees_brutes", VERSION, nom_video)
        os.makedirs(data_dir, exist_ok=True)
        
        print(f"\n[{nom_video}] ⚡ Extraction FFmpeg et analyse...")
        
        # 1. Extraction Audio via FFmpeg
        subprocess.run([
            "ffmpeg", "-y", "-i", video_path, 
            "-vn", "-acodec", "pcm_s16le", TEMP_WAV
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        audio_data, original_sr = librosa.load(TEMP_WAV, sr=None, mono=True)
        duree_totale = math.floor(len(audio_data) / original_sr)
        nb_windows = int(duree_totale // WINDOW_SIZE)

        if os.path.exists(TEMP_WAV):
            os.remove(TEMP_WAV)

        # 2. Chargement Vidéo
        cap = cv2.VideoCapture(video_path)
        fps_source = cap.get(cv2.CAP_PROP_FPS)

        # 3. Découpage Temporel
        for w in range(nb_windows):
            t_start = w * WINDOW_SIZE
            t_end = t_start + WINDOW_SIZE
            
            # --- AUDIO ---
            start_sample = int(t_start * original_sr)
            end_sample = int(t_end * original_sr)
            audio_chunk = audio_data[start_sample:end_sample]
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            
            chemin_audio = os.path.join(data_dir, f"audio_{t_start:.1f}_{t_end:.1f}.wav")
            wav.write(chemin_audio, original_sr, audio_int16)

            # --- VIDÉO ---
            dossier_frames = os.path.join(data_dir, f"frames_{t_start:.1f}_{t_end:.1f}")
            os.makedirs(dossier_frames, exist_ok=True)
            
            frames_a_lire = int(WINDOW_SIZE * fps_source)
            saved_frames = 0
            
            for _ in range(frames_a_lire):
                ret, frame = cap.read()
                if not ret: break
                nom_frame = os.path.join(dossier_frames, f"frame_{saved_frames:04d}.jpg")
                cv2.imwrite(nom_frame, frame)
                saved_frames += 1
                
        cap.release()
        print(f"✅ [{nom_video}] Terminé : {nb_windows} tranches sauvegardées.")

    print(f"\n🎉 Ingestion Batch terminée avec succès !")

if __name__ == "__main__":
    extraire_donnees_batch()