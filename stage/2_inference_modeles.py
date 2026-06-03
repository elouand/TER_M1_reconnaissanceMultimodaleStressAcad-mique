# -*- coding: utf-8 -*-
import os
import csv
import cv2
import numpy as np
import librosa
import math
import torch
import subprocess
from tqdm import tqdm

# Import de l'architecture IA
from audToVAD import get_acoustic_vad, stt_model, nlp_model, nlp_tokenizer, device_cpu
from traitementVideo import EmotionRegressor

# ==========================================
# CONFIGURATION
# ==========================================
VERSION = "v1.0"
NOM_VIDEO = "sujet_recola_test"
VIDEO_PATH = f"{NOM_VIDEO}.mp4" 
WINDOW_SIZE = 2.0

# Dossiers d'entrée et de sortie
DATA_DIR = os.path.join("donnees_brutes", VERSION, NOM_VIDEO)
CSV_DIR = os.path.join("adj_csv", VERSION)
CSV_OUT = os.path.join(CSV_DIR, f"{NOM_VIDEO}_inferred.csv")

# Création du dossier CSV s'il n'existe pas
os.makedirs(CSV_DIR, exist_ok=True)

def analyser_texte_global():
    """ 
    Analyse l'intégralité de l'audio avec Whisper et DistilBERT d'un seul coup.
    Retourne une liste de segments avec leurs scores VA.
    """
    print("\n📝 [TEXTE] Lancement de l'analyse STT globale sur la vidéo...")
    temp_wav = "audio_temp_inference.wav"
    
    # Extraction FFmpeg rapide et conversion directe en 16kHz
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", VIDEO_PATH, 
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", temp_wav
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        print("❌ Erreur : FFmpeg a échoué lors de l'extraction audio globale.")
        return [], 0.0

    # Chargement instantané depuis le WAV
    audio_full, _ = librosa.load(temp_wav, sr=16000, mono=True)
    
    # Nettoyage
    if os.path.exists(temp_wav):
        os.remove(temp_wav)
    
    # 1. Transcription globale avec Whisper
    result = stt_model.transcribe(
        audio_full, 
        language="fr", 
        fp16=False,
        initial_prompt="Ceci est une description d'image en français."
    )
    
    segments_annotes = []
    
    # 2. Analyse NLP (DistilBERT) sur chaque phrase trouvée
    print(f"🧠 [NLP] {len(result['segments'])} phrases détectées. Calcul des scores VA...")
    for seg in result["segments"]:
        texte = seg["text"].strip()
        if not texte:
            continue
            
        inputs = nlp_tokenizer(texte, return_tensors="pt", truncation=True, padding=True).to(device_cpu)
        with torch.no_grad():
            outputs = nlp_model(**inputs)
            
        scores = outputs.logits.squeeze().cpu().tolist()
        if isinstance(scores, float):
            scores = [scores, 0.0, 0.0]
            
        v_texte = max(-1.0, min(1.0, scores[0] * 2.5))
        a_texte = max(-1.0, min(1.0, scores[1] * 2.5))
        
        segments_annotes.append({
            "start": seg["start"],
            "end": seg["end"],
            "texte": texte,
            "v": v_texte,
            "a": a_texte
        })
        print(f"   -> [{seg['start']:.1f}s - {seg['end']:.1f}s] : '{texte}' (V:{v_texte:.2f}, A:{a_texte:.2f})")
        
    return segments_annotes, len(audio_full) / 16000.0

def run_inference():
    print(f"--- 🧠 DÉBUT DE L'INFÉRENCE (Version : {VERSION}) ---")
    
    if not os.path.exists(DATA_DIR):
        print(f"❌ Erreur : Le dossier de données brutes {DATA_DIR} n'existe pas. Lancez le script 1b d'abord.")
        return

    # 1. INITIALISATION DES MODÈLES VISUELS
    print("⏳ Chargement du modèle de Vision (HSEmotion)...")
    vision_model = EmotionRegressor(device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # 2. ANALYSE GLOBALE DU TEXTE ET RÉCUPÉRATION DE LA DURÉE
    phrases_annotees, duree_totale_audio = analyser_texte_global()
    
    # Calcul du nombre de fenêtres
    nb_windows = int(math.floor(duree_totale_audio) // WINDOW_SIZE)

    # 3. PRÉPARATION DU CSV
    headers = [
        "timestamp_start", "timestamp_end", 
        "dossier_frames", "chemin_audio", 
        "v_vision", "a_vision", 
        "v_audio", "a_audio", 
        "texte_transcrit", "v_texte", "a_texte", 
        "v_fusion", "a_fusion",
        "target_v", "target_a"
    ]
    lignes_out = []

    # 4. TRAITEMENT TRANCHE PAR TRANCHE
    print(f"\n🚀 Lancement de l'analyse multimodale sur {nb_windows} tranches...")
    
    for w in tqdm(range(nb_windows), desc="Analyse Vision & Ton", unit="tranche"):
        t_start = w * WINDOW_SIZE
        t_end = t_start + WINDOW_SIZE
        
        dossier_frames = os.path.join(DATA_DIR, f"frames_{t_start:.1f}_{t_end:.1f}")
        chemin_audio = os.path.join(DATA_DIR, f"audio_{t_start:.1f}_{t_end:.1f}.wav")
        
        row = {
            "timestamp_start": f"{t_start:.1f}",
            "timestamp_end": f"{t_end:.1f}",
            "dossier_frames": dossier_frames,
            "chemin_audio": chemin_audio,
            "v_fusion": "", "a_fusion": "",
            "target_v": "", "target_a": ""
        }
        
        # --- A. MAPPING DU TEXTE ---
        textes_tranche = []
        v_textes, a_textes = [], []
        
        for phrase in phrases_annotees:
            if (phrase["start"] < t_end) and (phrase["end"] > t_start):
                textes_tranche.append(phrase["texte"])
                v_textes.append(phrase["v"])
                a_textes.append(phrase["a"])
                
        if textes_tranche:
            row["texte_transcrit"] = " | ".join(textes_tranche)
            row["v_texte"] = round(float(np.mean(v_textes)), 4)
            row["a_texte"] = round(float(np.mean(a_textes)), 4)
        else:
            row["texte_transcrit"] = "[SILENCE]"
            row["v_texte"] = 0.0
            row["a_texte"] = 0.0

        # --- B. ANALYSE AUDIO (Ton) ---
        if os.path.exists(chemin_audio):
            audio_data, sr = librosa.load(chemin_audio, sr=16000, mono=True)
            res_aud = get_acoustic_vad(audio_data, sampling_rate=16000)
            if res_aud:
                row["v_audio"] = round(float((res_aud['valence'] * 2) - 1), 4)
                row["a_audio"] = round(float((res_aud['arousal'] * 2) - 1), 4)
            else:
                row["v_audio"], row["a_audio"] = 0.0, 0.0
        else:
            row["v_audio"], row["a_audio"] = 0.0, 0.0

        # --- C. ANALYSE VISION (HSEmotion) ---
        v_list, a_list = [], []
        
        if os.path.exists(dossier_frames):
            images = sorted(os.listdir(dossier_frames))
            for img_name in images:
                img_path = os.path.join(dossier_frames, img_name)
                frame = cv2.imread(img_path)
                
                if frame is not None:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    va_v = vision_model.get_vad(frame_rgb)
                    if va_v is not None:
                        v_list.append(va_v[0])
                        a_list.append(va_v[1])
            
            if len(v_list) > 0:
                row["v_vision"] = round(float(np.mean(v_list)), 4)
                row["a_vision"] = round(float(np.mean(a_list)), 4)
            else:
                row["v_vision"], row["a_vision"] = 0.0, 0.0
        else:
            row["v_vision"], row["a_vision"] = 0.0, 0.0
            
        lignes_out.append(row)

    # 5. SAUVEGARDE DES RÉSULTATS
    with open(CSV_OUT, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(lignes_out)

    print(f"\n🎉 Inférence terminée avec succès ! -> {CSV_OUT}")

if __name__ == "__main__":
    run_inference()