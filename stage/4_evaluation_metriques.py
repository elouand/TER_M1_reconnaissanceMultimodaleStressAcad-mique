# -*- coding: utf-8 -*-
import os
import csv
import numpy as np
import glob

# ==========================================
# CONFIGURATION
# ==========================================
VERSION = "v1.0"
CSV_DIR = os.path.join("adj_csv", VERSION)

def calculer_ccc(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) == 0 or len(y_pred) == 0 or np.var(y_true) == 0 or np.var(y_pred) == 0:
        return 0.0
    cor = np.corrcoef(y_true, y_pred)[0][1]
    numerator = 2 * cor * np.std(y_true) * np.std(y_pred)
    denominator = np.var(y_true) + np.var(y_pred) + (np.mean(y_true) - np.mean(y_pred))**2
    return numerator / denominator

def evaluer_performances_batch():
    print(f"--- 📈 ÉVALUATION SCIENTIFIQUE DU DEV SET (Version : {VERSION}) ---\n")
    
    csv_fused = glob.glob(os.path.join(CSV_DIR, "*_fused.csv"))
    if not csv_fused:
        return print("❌ Aucun fichier *_fused.csv trouvé.")

    # Pour le calcul du score global (concaténation de toutes les vidéos)
    all_targets_v, all_targets_a = [], []
    all_preds_v, all_preds_a = [], []
    
    scores_individuels = []

    for csv_in in csv_fused:
        nom_base = os.path.basename(csv_in).replace("_fused.csv", "")
        t_v, t_a, p_v, p_a = [], [], [], []
        
        with open(csv_in, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("target_v") not in [None, ""] and row.get("target_a") not in [None, ""]:
                    t_v.append(float(row["target_v"]))
                    t_a.append(float(row["target_a"]))
                    p_v.append(float(row["v_fusion"]))
                    p_a.append(float(row["a_fusion"]))
                    
        # Ajouter aux listes globales
        all_targets_v.extend(t_v)
        all_targets_a.extend(t_a)
        all_preds_v.extend(p_v)
        all_preds_a.extend(p_a)
        
        # Calcul individuel
        if t_v:
            ccc_v = calculer_ccc(t_v, p_v)
            ccc_a = calculer_ccc(t_a, p_a)
            scores_individuels.append({"video": nom_base, "ccc_v": ccc_v, "ccc_a": ccc_a})

    # Affichage des scores par vidéo
    print("📋 RÉSULTATS INDIVIDUELS :")
    for s in scores_individuels:
        print(f" - {s['video'][:15]:<15} | CCC_V: {s['ccc_v']:.4f} | CCC_A: {s['ccc_a']:.4f}")

    # Affichage du score GLOBAL
    if all_targets_v:
        global_ccc_v = calculer_ccc(all_targets_v, all_preds_v)
        global_ccc_a = calculer_ccc(all_targets_a, all_preds_a)
        
        print("\n" + "=" * 50)
        print("🏆 SCORE MOYEN GLOBAL (DEV SET)")
        print("=" * 50)
        print(f"🔹 CCC Valence : {global_ccc_v:.4f}")
        print(f"🔸 CCC Arousal : {global_ccc_a:.4f}")
        print("=" * 50)
    else:
        print("\n⚠️ Attention : Aucune valeur 'target' trouvée. Impossible de calculer l'erreur.")

if __name__ == "__main__":
    evaluer_performances_batch()