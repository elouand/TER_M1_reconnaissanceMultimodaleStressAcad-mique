# -*- coding: utf-8 -*-
import os
import csv
import cv2
import numpy as np
import librosa
import math
import torch
import torch.nn.functional as F  # <--- AJOUT INDISPENSABLE
import subprocess
import glob
from tqdm import tqdm

from audToVAD import get_acoustic_vad, stt_model, nlp_model, nlp_tokenizer, device_cpu
from traitementVideo import EmotionRegressor
                
from audToVAD import is_speech_present

# ==========================================
# CONFIGURATION
# ==========================================
VERSION = "v1.2" 
WINDOW_SIZE = 2.0
VIDEO_DIR = "videos_test"
CSV_DIR = os.path.join("adj_csv", VERSION)

os.makedirs(CSV_DIR, exist_ok=True)

def analyser_texte_global(video_path):
    temp_wav = "audio_temp_inference.wav"
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path, 
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", temp_wav
    ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    audio_full, _ = librosa.load(temp_wav, sr=16000, mono=True)
    if os.path.exists(temp_wav): os.remove(temp_wav)
    
    result = stt_model.transcribe(audio_full, language="fr", fp16=False)
    segments_annotes = []
    
    for seg in result["segments"]:
        texte = seg["text"].strip()
        if not texte: continue
            
        inputs = nlp_tokenizer(texte, return_tensors="pt", truncation=True, max_length=512).to(device_cpu)
        with torch.no_grad():
            outputs = nlp_model(**inputs)
            
        # ========================================================
        # NOUVELLE LOGIQUE MATHÉMATIQUE (Softmax + Espérance)
        # ========================================================
        probs = F.softmax(outputs.logits, dim=-1).squeeze().cpu().tolist()
        if isinstance(probs, float): probs = [probs]

        # Mapping : 1 étoile (-1.0), 2 étoiles (-0.5), ..., 5 étoiles (1.0)
        poids_valence = [-1.0, -0.5, 0.0, 0.5, 1.0]
        if len(probs) == 5:
            v_texte = sum(p * w for p, w in zip(probs, poids_valence))
        else:
            v_texte = 0.0
            
        a_texte = abs(v_texte) * 0.8
        # ========================================================
        
        segments_annotes.append({"start": seg["start"], "end": seg["end"], "texte": texte, "v": v_texte, "a": a_texte})
        
    return segments_annotes, len(audio_full) / 16000.0

def run_inference_batch():
    print(f"--- 🧠 DÉBUT DE L'INFÉRENCE BATCH (Version : {VERSION}) ---")
    
    videos = glob.glob(os.path.join(VIDEO_DIR, "*.mp4"))
    if not videos: return print("❌ Aucune vidéo trouvée.")

    print("⏳ Chargement unique du modèle de Vision (HSEmotion)...")
    vision_model = EmotionRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')

    for video_path in videos:
        nom_video = os.path.splitext(os.path.basename(video_path))[0]
        data_dir = os.path.join("donnees_brutes", VERSION, nom_video)
        csv_out = os.path.join(CSV_DIR, f"{nom_video}_inferred.csv")

        if not os.path.exists(data_dir):
            print(f"⚠️ Données brutes manquantes pour {nom_video}. Avez-vous bien lancé 1b_ingestion en v1.1 ?")
            continue

        print(f"\n📝 [{nom_video}] STT & NLP Global...")
        phrases_annotees, duree_totale = analyser_texte_global(video_path)
        nb_windows = int(math.floor(duree_totale) // WINDOW_SIZE)

        headers = ["timestamp_start", "timestamp_end", "dossier_frames", "chemin_audio", 
                   "v_vision", "a_vision", "v_audio", "a_audio", "texte_transcrit", 
                   "v_texte", "a_texte", "v_fusion", "a_fusion", "target_v", "target_a"]
        lignes_out = []

        for w in tqdm(range(nb_windows), desc=f"Analyse {nom_video}", unit="tranche"):
            t_start, t_end = w * WINDOW_SIZE, (w + 1) * WINDOW_SIZE
            dossier_frames = os.path.join(data_dir, f"frames_{t_start:.1f}_{t_end:.1f}")
            chemin_audio = os.path.join(data_dir, f"audio_{t_start:.1f}_{t_end:.1f}.wav")
            
            row = {"timestamp_start": f"{t_start:.1f}", "timestamp_end": f"{t_end:.1f}",
                   "dossier_frames": dossier_frames, "chemin_audio": chemin_audio,
                   "v_fusion": "", "a_fusion": "", "target_v": "", "target_a": ""}
            
            # NLP
            v_textes, a_textes, textes_tranche = [], [], []
            for p in phrases_annotees:
                if (p["start"] < t_end) and (p["end"] > t_start):
                    textes_tranche.append(p["texte"])
                    v_textes.append(p["v"])
                    a_textes.append(p["a"])
                    
            if textes_tranche:
                row["texte_transcrit"] = " | ".join(textes_tranche)
                row["v_texte"] = round(float(np.mean(v_textes)), 4)
                row["a_texte"] = round(float(np.mean(a_textes)), 4)
            else:
                row["texte_transcrit"], row["v_texte"], row["a_texte"] = "[SILENCE]", 0.0, 0.0

            # AUDIO
            if os.path.exists(chemin_audio):
                audio_data, sr = librosa.load(chemin_audio, sr=16000, mono=True)
                if is_speech_present(audio_data, sampling_rate=16000):
                    # Il y a une voix ! On lance l'analyse d'émotion Wav2Vec2
                    res_aud = get_acoustic_vad(audio_data, sampling_rate=16000)
                    if res_aud:
                        row["v_audio"] = round(float((res_aud['valence'] * 2) - 1), 4)
                        row["a_audio"] = round(float((res_aud['arousal'] * 2) - 1), 4)
                    else: 
                        row["v_audio"], row["a_audio"] = 0.0, 0.0
                else:
                    # SILENCE ACOUSTIQUE DÉTECTÉ : On coupe les valeurs parasites !
                    row["v_audio"], row["a_audio"] = 0.0, 0.0
            else: 
                row["v_audio"], row["a_audio"] = 0.0, 0.0

            # VISION
            v_list, a_list = [], []
            if os.path.exists(dossier_frames):
                images = sorted(os.listdir(dossier_frames))
                for img_name in images:
                    frame = cv2.imread(os.path.join(dossier_frames, img_name))
                    if frame is not None:
                        va_v = vision_model.get_vad(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        if va_v is not None:
                            v_list.append(va_v[0])
                            a_list.append(va_v[1])
                if v_list:
                    row["v_vision"] = round(float(np.mean(v_list)), 4)
                    row["a_vision"] = round(float(np.mean(a_list)), 4)
                else: row["v_vision"], row["a_vision"] = 0.0, 0.0
            else: row["v_vision"], row["a_vision"] = 0.0, 0.0
                
            lignes_out.append(row)

        with open(csv_out, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(lignes_out)

    print("\n🎉 Inférence Batch terminée !")

if __name__ == "__main__":
    run_inference_batch()