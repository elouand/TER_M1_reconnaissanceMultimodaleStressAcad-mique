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

WEIGHTS = {
    "vision": np.array([0.8, 0.4]),
    "audio":  np.array([0.4, 0.9]),
    "texte":  np.array([0.9, 0.3]) 
}

def calculer_fusion_batch():
    print(f"--- 🧬 DÉBUT DE LA FUSION BATCH (Version : {VERSION}) ---")
    
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

        for row in lignes_in:
            def get_val(cle):
                return float(row[cle]) if row.get(cle) not in [None, ""] else 0.0

            v_v = np.array([get_val("v_vision"), get_val("a_vision")])
            v_a = np.array([get_val("v_audio"), get_val("a_audio")])
            v_t = np.array([get_val("v_texte"), get_val("a_texte")])

            num = (v_v * WEIGHTS["vision"]) + (v_a * WEIGHTS["audio"]) + (v_t * WEIGHTS["texte"])
            den = WEIGHTS["vision"] + WEIGHTS["audio"] + WEIGHTS["texte"]

            fusion = np.clip(num / (den + 1e-6), -1.0, 1.0)
            row["v_fusion"] = round(float(fusion[0]), 4)
            row["a_fusion"] = round(float(fusion[1]), 4)

        with open(csv_out, mode='w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(lignes_in)

        print(f"✅ Fusion appliquée pour : {nom_base}")

    print("🎉 Fusion Batch terminée !")

if __name__ == "__main__":
    calculer_fusion_batch()