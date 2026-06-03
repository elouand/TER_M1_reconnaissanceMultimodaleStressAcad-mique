# -*- coding: utf-8 -*-
import os
import csv
import numpy as np

# ==========================================
# CONFIGURATION
# ==========================================
VERSION = "v1.0"
NOM_VIDEO = "sujet_recola_test"
CSV_FINAL = os.path.join("adj_csv", VERSION, f"{NOM_VIDEO}_fused.csv")

def calculer_ccc(y_true, y_pred):
    """
    Calcule le Coefficient de Corrélation de Concordance (CCC).
    Contrairement à la MSE (Erreur Quadratique Moyenne), le CCC pénalise 
    non seulement la variance (bruit), mais aussi les biais de décalage 
    entre les prédictions et la vérité terrain.
    Plage : -1 (inversion totale) à 1 (accord parfait).
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Si les tableaux sont vides ou constants (variance nulle)
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0
    if np.var(y_true) == 0 or np.var(y_pred) == 0:
        return 0.0

    # Moyennes et Variances
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    
    # Ecart-types
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)
    
    # Corrélation de Pearson
    cor = np.corrcoef(y_true, y_pred)[0][1]
    
    # Formule du CCC
    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred)**2
    
    return numerator / denominator

def evaluer_performances():
    print(f"--- 📈 DÉBUT DE L'ÉVALUATION SCIENTIFIQUE (Version : {VERSION}) ---")
    
    if not os.path.exists(CSV_FINAL):
        print(f"❌ Erreur : Le fichier final {CSV_FINAL} n'existe pas.")
        return

    # Initialisation des listes pour stocker les séries temporelles
    targets_v, targets_a = [], []
    preds_v, preds_a = [], []
    
    # Lecture du CSV
    with open(CSV_FINAL, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # On vérifie que la ligne contient bien les données cibles de RECOLA
            if row["target_v"] != "" and row["target_a"] != "":
                targets_v.append(float(row["target_v"]))
                targets_a.append(float(row["target_a"]))
                preds_v.append(float(row["v_fusion"]))
                preds_a.append(float(row["a_fusion"]))

    if not targets_v:
        print("⚠️ Attention : Aucune valeur 'target_v' ou 'target_a' trouvée dans le CSV.")
        print("-> Ce script est prêt, mais il attend que vous intégriez les annotations RECOLA.")
        return

    # Calcul des métriques
    ccc_valence = calculer_ccc(targets_v, preds_v)
    ccc_arousal = calculer_ccc(targets_a, preds_a)
    
    # Affichage formaté pour le rapport de TER
    print("\n📊 RÉSULTATS DES PERFORMANCES (Concordance Correlation Coefficient) :")
    print("-" * 50)
    print(f"🔹 CCC Valence : {ccc_valence:.4f}")
    print(f"🔸 CCC Arousal : {ccc_arousal:.4f}")
    print("-" * 50)
    
    # Interprétation grossière pour aider l'analyse
    print("\n💡 Interprétation :")
    for axe, score in [("Valence", ccc_valence), ("Arousal", ccc_arousal)]:
        if score > 0.8:
            qualite = "Excellent (Niveau État de l'Art)"
        elif score > 0.5:
            qualite = "Bon (Le modèle capte la tendance générale)"
        elif score > 0.2:
            qualite = "Faible (Bruit important ou biais systémique détecté)"
        else:
            qualite = "Échec (Pas de corrélation avec la vérité terrain)"
        print(f"- {axe} : {qualite}")

if __name__ == "__main__":
    evaluer_performances()