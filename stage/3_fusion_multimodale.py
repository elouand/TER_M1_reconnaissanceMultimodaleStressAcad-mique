# -*- coding: utf-8 -*-
import os
import csv
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
VERSION = "v1.0"
NOM_VIDEO = "sujet_recola_test"
CSV_IN = os.path.join("adj_csv", VERSION, f"{NOM_VIDEO}_inferred.csv")
CSV_OUT = os.path.join("adj_csv", VERSION, f"{NOM_VIDEO}_fused.csv")

# MATRICE DE POIDS VA (Tirée de ton MultimodalState.py)
# Format : [Poids_Valence, Poids_Arousal]
WEIGHTS = {
    "vision": np.array([0.8, 0.4]), # Expert Valence (Sourire)
    "audio":  np.array([0.4, 0.9]), # Expert Arousal (Stress/Volume)
    "texte":  np.array([0.9, 0.3])  # Expert Valence (Mots-clés)
}

def calculer_fusion():
    print(f"--- 🧬 DÉBUT DE LA FUSION MULTIMODALE (Version : {VERSION}) ---")
    
    if not os.path.exists(CSV_IN):
        print(f"❌ Erreur : Le fichier d'inférence {CSV_IN} n'existe pas.")
        return

    lignes_in = []
    with open(CSV_IN, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            lignes_in.append(row)

    print(f"🚀 Application de la matrice de poids sur {len(lignes_in)} tranches...")

    for row in lignes_in:
        # 1. Récupération des valeurs (avec sécurité si la case est vide)
        def get_val(cle):
            return float(row[cle]) if row[cle] != "" else 0.0

        v_v = np.array([get_val("v_vision"), get_val("a_vision")])
        v_a = np.array([get_val("v_audio"), get_val("a_audio")])
        v_t = np.array([get_val("v_texte"), get_val("a_texte")])

        # 2. Calcul pondéré VA (Formule exacte de ton code Naoqi)
        num = (v_v * WEIGHTS["vision"]) + \
              (v_a * WEIGHTS["audio"]) + \
              (v_t * WEIGHTS["texte"])
              
        den = WEIGHTS["vision"] + WEIGHTS["audio"] + WEIGHTS["texte"]

        # 3. Normalisation (entre -1.0 et 1.0)
        fusion = np.clip(num / (den + 1e-6), -1.0, 1.0)

        # 4. Écriture dans la ligne
        row["v_fusion"] = round(float(fusion[0]), 4)
        row["a_fusion"] = round(float(fusion[1]), 4)

    # Sauvegarde du CSV Final
    with open(CSV_OUT, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lignes_in)

    print(f"✅ Fusion pondérée appliquée avec succès !")
    print(f"📄 Fichier final prêt pour l'évaluation : {CSV_OUT}")

if __name__ == "__main__":
    calculer_fusion()