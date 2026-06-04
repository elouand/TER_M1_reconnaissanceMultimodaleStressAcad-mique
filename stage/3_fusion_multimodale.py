# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import glob

# ==========================================
# CONFIGURATION
# ==========================================
VERSION = "v1.1"
CSV_DIR = os.path.join("adj_csv", VERSION)

# Poids de base (seront modifiés dynamiquement)
BASE_WEIGHTS = {
    "vision": {"v": 0.8, "a": 0.4},
    "audio":  {"v": 0.4, "a": 0.9},
    "texte":  {"v": 0.9, "a": 0.3} 
}

# Paramètres de Lissage (Rémanence Émotionnelle)
# 1.0 = Aucun lissage (réactif 100%) | 0.0 = Totalement figé
# 0.6 = Bon équilibre entre réactivité et mémoire à court terme
ALPHA_V = 0.6 
ALPHA_A = 0.6

def calculer_fusion_batch():
    print(f"--- 🧬 DÉBUT DE LA FUSION BATCH DYNAMIQUE (Version : {VERSION}) ---")
    
    csv_inferred = glob.glob(os.path.join(CSV_DIR, "*_inferred.csv"))
    if not csv_inferred:
        print("❌ Aucun fichier *_inferred.csv trouvé.")
        return

    for csv_in in csv_inferred:
        nom_base = os.path.basename(csv_in).replace("_inferred.csv", "")
        csv_out = os.path.join(CSV_DIR, f"{nom_base}_fused.csv")
        
        lignes_in = []
        with open(csv_in, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                lignes_in.append(row)

        # Variables pour garder en mémoire l'état émotionnel précédent (EMA)
        prev_fusion_v = 0.0
        prev_fusion_a = 0.0
        is_first_row = True

        for row in lignes_in:
            def get_val(cle):
                return float(row[cle]) if row.get(cle) not in [None, ""] else 0.0

            # Extraction des données brutes
            v_v, a_v = get_val("v_vision"), get_val("a_vision")
            v_a, a_a = get_val("v_audio"), get_val("a_audio")
            v_t, a_t = get_val("v_texte"), get_val("a_texte")
            texte_transcrit = row.get("texte_transcrit", "")

            # =========================================================
            # 1. PONDÉRATION DYNAMIQUE (Mécanisme d'Attention)
            # =========================================================
            # Si la vision retourne exactement 0.0 partout, c'est qu'il n'y a pas de visage
            visage_absent = (v_v == 0.0 and a_v == 0.0)
            w_vision_v = 0.0 if visage_absent else BASE_WEIGHTS["vision"]["v"]
            w_vision_a = 0.0 if visage_absent else BASE_WEIGHTS["vision"]["a"]
            
            # Si c'est un silence, le texte ne compte plus
            silence = (texte_transcrit == "[SILENCE]")
            w_texte_v = 0.0 if silence else BASE_WEIGHTS["texte"]["v"]
            w_texte_a = 0.0 if silence else BASE_WEIGHTS["texte"]["a"]
            
            # L'audio (le ton) est toujours pertinent, même en cas de silence
            w_audio_v = BASE_WEIGHTS["audio"]["v"] 
            w_audio_a = BASE_WEIGHTS["audio"]["a"]

            # =========================================================
            # 2. CALCUL INSTANTANÉ
            # =========================================================
            num_v = (v_v * w_vision_v) + (v_a * w_audio_v) + (v_t * w_texte_v)
            den_v = w_vision_v + w_audio_v + w_texte_v
            inst_fusion_v = num_v / (den_v + 1e-6) # 1e-6 évite la division par zéro

            num_a = (a_v * w_vision_a) + (a_a * w_audio_a) + (a_t * w_texte_a)
            den_a = w_vision_a + w_audio_a + w_texte_a
            inst_fusion_a = num_a / (den_a + 1e-6)

            # =========================================================
            # 3. LISSAGE TEMPOREL (Rémanence EMA)
            # =========================================================
            if is_first_row:
                final_v = inst_fusion_v
                final_a = inst_fusion_a
                is_first_row = False
            else:
                final_v = (ALPHA_V * inst_fusion_v) + ((1 - ALPHA_V) * prev_fusion_v)
                final_a = (ALPHA_A * inst_fusion_a) + ((1 - ALPHA_A) * prev_fusion_a)

            # Mise à jour de la mémoire pour la boucle suivante
            prev_fusion_v = final_v
            prev_fusion_a = final_a

            # Sauvegarde propre avec limitation stricte entre -1.0 et 1.0
            row["v_fusion"] = round(float(np.clip(final_v, -1.0, 1.0)), 4)
            row["a_fusion"] = round(float(np.clip(final_a, -1.0, 1.0)), 4)

        # Écriture du fichier fusionné
        with open(csv_out, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(lignes_in)

        print(f"✅ Fusion dynamique & lissée appliquée pour : {nom_base}")

    print("🎉 Fusion Batch v1.1 terminée avec succès !")

if __name__ == "__main__":
    calculer_fusion_batch()