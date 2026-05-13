# -*- coding: utf-8 -*-
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import librosa
import whisper
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

import torch
torch.cuda.empty_cache() # Vide les restes de mémoire des anciens crashs

# --- ARCHITECTURE OFFICIELLE AUDEERING ---
class RegressionHead(nn.Module):
    r"""Classification head."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.final_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.classifier = RegressionHead(config)
        
        # FIX COMPATIBILITÉ
        self.all_tied_weights_keys = {} 
        self.init_weights()

    def forward(self, input_values):
        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        logits = self.classifier(hidden_states)
        return hidden_states, logits

# --- INITIALISATION DES APPAREILS ---
device = "cuda" if torch.cuda.is_available() else "cpu"
device_cpu = "cpu" # On force le CPU pour les modèles lourds mais non-temps-réel

# 1. Modèle de Ton (Acoustique) -> CUDA (Carte Graphique)
PATH_LOCAL = "./modele_ton"
print("Chargement du modèle local audEERING (Wav2Vec2-Large-Robust)...")
processor = Wav2Vec2Processor.from_pretrained(PATH_LOCAL)
model = EmotionModel.from_pretrained(PATH_LOCAL).to(device)
model.eval()

# 2. Modèle STT (Whisper) -> CPU
print("Chargement du modèle STT (Whisper)...")
# Ajout de device=device_cpu ici
stt_model = whisper.load_model("base", device=device_cpu)

# 3. Modèle NLP (DistilBERT) -> CPU
print("Chargement du modèle NLP (DistilBERT)...")
PATH_NLP = "./modele_texte"
nlp_tokenizer = AutoTokenizer.from_pretrained(PATH_NLP)
# Remplacement de .to(device) par .to(device_cpu) ici
nlp_model = AutoModelForSequenceClassification.from_pretrained(PATH_NLP).to(device_cpu)
nlp_model.eval()


def get_acoustic_vad(audio_numpy, sampling_rate=16000):
    """
    Predict emotions: arousal, dominance, valence (0...1).
    """
    try:
        # 1. Conversion float32 normalisé
        if audio_numpy.dtype == np.int16:
            audio_float = audio_numpy.astype(np.float32) / 32768.0
        else:
            audio_float = audio_numpy

        # 2. Resampling PHYSIQUE -> 16k
        audio_16k = librosa.resample(audio_float, orig_sr=sampling_rate, target_sr=16000)

        # 3. Passage au processor
        y = processor(audio_16k, sampling_rate=16000, return_tensors="pt", padding=True)
        y = y['input_values'].to(device)

        # 4. Inférence protégée pour éviter de saturer la RAM de la carte graphique
        with torch.no_grad():
            _, logits = model(y)

        scores = logits.detach().cpu().numpy()[0] 
        
        return {
            'arousal': float(scores[0]),
            'dominance': float(scores[1]),
            'valence': float(scores[2])
        }
    except Exception as e:
        print(f"Erreur VAD acoustique : {e}")
        return None
    

def get_text_vad(audio_segment, orig_sr=16000):
    try:
        # 1. Resampling
        audio_16k = librosa.resample(audio_segment, orig_sr=orig_sr, target_sr=16000)

        # 2. Normalisation stricte pour Whisper
        max_val = np.max(np.abs(audio_16k))
        if max_val > 0:
            audio_16k = audio_16k / max_val

        # 3. Inférence STT (Whisper) - S'exécute sur le CPU
        result = stt_model.transcribe(
            audio_16k, 
            language="fr", 
            fp16=False, # fp16 doit être False sur CPU
            initial_prompt="Ceci est une description d'image en français."
        )
        
        # ---> C'est ici que "texte" est défini <---
        texte = result["text"].strip() 
        
        if not texte:
            return None, [0.0, 0.0, 0.0]
            
        # 4. Inférence NLP (DistilBERT)
        # ---> C'est ici que "inputs" est défini <---
        inputs = nlp_tokenizer(texte, return_tensors="pt", truncation=True, padding=True).to(device_cpu)
        
        with torch.no_grad():
            outputs = nlp_model(**inputs)
            
        scores = outputs.logits.squeeze().cpu().tolist()
        
        if isinstance(scores, float):
            scores = [scores, 0.0, 0.0]
            
        # =========================================================
        # EXACERBATION (SCALING) DES SCORES NLP
        # =========================================================
        FACTEUR = 2.5
        
        scores_etires = []
        for s in scores:
            scores_etires.append(max(-1.0, min(1.0, s * FACTEUR)))
            
        # Sécurité : Si le modèle ne renvoie que 2 scores (V, A), 
        # on ajoute un 0.0 pour la dominance afin de respecter le format attendu.
        while len(scores_etires) < 3:
            scores_etires.append(0.0)
        
        return texte, scores_etires
        # =========================================================
        
    except Exception as e:
        print(f"Erreur STT/NLP détaillée : {e}")
        return None, None

if __name__ == "__main__":
    # Test silence
    test_signal = np.zeros(16000, dtype=np.float32)
    print("Test silence (Ton) :", get_acoustic_vad(test_signal, sampling_rate=16000))
    print("Test silence (Texte) :", get_text_vad(test_signal, orig_sr=16000))