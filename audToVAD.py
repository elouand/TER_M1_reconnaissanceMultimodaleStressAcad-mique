# -*- coding: utf-8 -*-
import torch
import numpy as np
import librosa
from huggingface_hub import login
from transformers import pipeline

# 1. Authentification
MON_TOKEN = "token dans le .env"
login(token=MON_TOKEN)

# 2. Chargement du modèle
# Note : Le premier lancement téléchargera environ 1.2 Go de données
print("Initialisation du modèle VAD (Audeering Wav2Vec2)...")

try:
    classifier = pipeline(
        "audio-classification", 
        model="audeering/wav2vec2-large-robust-12-ft-emotion-msp-podcast",
        token=MON_TOKEN
    )
    print("Modèle prêt et mis en cache !")
except Exception as e:
    print("Erreur lors du chargement du modèle : {0}".format(e))

def get_acoustic_vad(audio_numpy, sampling_rate=48000):
    """
    Prend un tableau numpy d'audio (int16) et renvoie les scores VAD.
    Conversion et rééchantillonnage inclus pour la compatibilité Pepper -> IA.
    """
    try:
        # A. Conversion int16 (Pepper) -> float32 normalisé [-1, 1]
        if audio_numpy.dtype == np.int16:
            audio_float = audio_numpy.astype(np.float32) / 32768.0
        else:
            audio_float = audio_numpy

        # B. Rééchantillonnage de 48kHz (Pepper) vers 16kHz (IA)
        # Indispensable pour ne pas fausser l'analyse du ton
        audio_16k = librosa.resample(audio_float, orig_sr=sampling_rate, target_sr=16000)

        # C. Inférence
        results = classifier(audio_16k)
        
        # Le modèle renvoie 'arousal', 'dominance', 'valence'
        vad_dict = {res['label']: res['score'] for res in results}
        return vad_dict

    except Exception as e:
        print("Erreur lors de l'inférence VAD : {0}".format(e))
        return None

if __name__ == "__main__":
    # Test rapide si lancé seul
    print("Test du script audToVAD...")
    # Simulation d'un second de silence à 48kHz
    test_signal = np.zeros(48000, dtype=np.int16)
    print("Résultat test (silence) :", get_acoustic_vad(test_signal))