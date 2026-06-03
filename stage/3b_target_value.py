# -*- coding: utf-8 -*-
import os
import csv
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
VERSION = "v1.0"
NOM_VIDEO = "sujet_recola_test"

# Fichiers sources
CSV_FUSION = os.path.join("adj_csv", VERSION, f"{NOM_VIDEO}_fused.csv")
DOSSIER_RECOLA = os.path.join("donnees_brutes", VERSION, "recola_annotations_simulees")
FICHIER_RECOLA_V = os.path.join(DOSSIER_RECOLA, "test_valence.csv")
FICHIER_RECOLA_A = os.path.join(DOSSIER_RECOLA, "test_arousal.csv")

def charger_annotations_recola(chemin_fichier):
    """ Lit le fichier RECOLA 25Hz et retourne une liste de tuples (temps, valeur) """
    donnees = []
    with open(chemin_fichier, mode='r') as f:
        # On tente de deviner le délimiteur (parfois ';' parfois ',')
        reader = csv.DictReader(f, delimiter=';' if ';' in f.readline() else ',')
        f.seek(0)
        reader.__next__() # Skip header
        for row in reader:
            donnees.append((float(row["time"]), float(row["value"])))
    return donnees

def injecter_cibles():
    print(f"--- 🎯 INJECTION DES CIBLES RECOLA (Version : {VERSION}) ---")
    
    if not os.path.exists(CSV_FUSION):
        print(f"❌ Erreur : Le fichier de fusion {CSV_FUSION} n'existe pas.")
        return

    # 1. Chargement de la haute fréquence (25 Hz)
    print("⏳ Chargement des annotations continues (Gold Standard)...")
    recola_v = charger_annotations_recola(FICHIER_RECOLA_V)
    recola_a = charger_annotations_recola(FICHIER_RECOLA_A)
    print(f"   -> {len(recola_v)} points d'annotation trouvés.")

    # 2. Lecture du CSV de ton pipeline (0.5 Hz)
    lignes_in = []
    with open(CSV_FUSION, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        for row in reader:
            lignes_in.append(row)

    # 3. Alignement temporel (Window Averaging)
    print("🔄 Alignement temporel et calcul des moyennes...")
    for row in lignes_in:
        t_start = float(row["timestamp_start"])
        t_end = float(row["timestamp_end"])
        
        # On extrait toutes les annotations RECOLA qui tombent dans notre fenêtre de 2 secondes
        valeurs_v_fenetre = [val for (t, val) in recola_v if t_start <= t < t_end]
        valeurs_a_fenetre = [val for (t, val) in recola_a if t_start <= t < t_end]
        
        # On calcule la moyenne, et on l'injecte dans la colonne cible
        if valeurs_v_fenetre:
            row["target_v"] = round(float(np.mean(valeurs_v_fenetre)), 4)
        if valeurs_a_fenetre:
            row["target_a"] = round(float(np.mean(valeurs_a_fenetre)), 4)

    # 4. Écrasement du fichier fusion avec les targets remplies
    with open(CSV_FUSION, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(lignes_in)

    print(f"✅ Injection réussie ! Le fichier {CSV_FUSION} est prêt pour l'évaluation.")
    print("-> Vous pouvez maintenant lancer le fichier '4_evaluation_metriques.py'.")

if __name__ == "__main__":
    injecter_cibles()