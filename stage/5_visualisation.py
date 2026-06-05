# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob

# ==========================================
# CONFIGURATION
# ==========================================
VERSION = "v1.2"
CSV_DIR = os.path.join("adj_csv", VERSION)

def tracer_graphiques(chemin_csv):
    nom_fichier = os.path.basename(chemin_csv).replace('.csv', '')
    print(f"📊 Génération des graphiques pour : {nom_fichier}")
    
    # 1. Chargement des données avec Pandas
    df = pd.read_csv(chemin_csv)
    
    # Création d'une colonne temps (milieu de la fenêtre)
    df['temps'] = (df['timestamp_start'] + df['timestamp_end']) / 2.0

    # 2. Configuration de la figure globale
    fig, (ax_v, ax_a) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Analyse Multimodale des Émotions ({VERSION})", fontsize=16, fontweight='bold')

    # ==========================================
    # GRAPHIQUE 1 : VALENCE
    # ==========================================
    ax_v.set_title("Évolution de la Valence", fontsize=14)
    ax_v.set_ylabel("Valence (-1.0 à 1.0)", fontsize=12)
    ax_v.set_ylim(-1.05, 1.05)
    ax_v.axhline(0, color='black', linewidth=1, linestyle='--') # Ligne de neutralité

    # Tracé des modalités (fines et semi-transparentes)
    ax_v.plot(df['temps'], df['v_vision'], label='Vision (ViT)', color='#3498db', linewidth=2, alpha=0.7)
    ax_v.plot(df['temps'], df['v_texte'], label='Texte (CamemBERT)', color='#2ecc71', linewidth=2, alpha=0.7)
    ax_v.plot(df['temps'], df['v_audio'], label='Audio/Ton (Wav2Vec2)', color='#f39c12', linewidth=2, alpha=0.7)
    
    # Tracé de la Fusion (Épaisse et bien visible)
    ax_v.plot(df['temps'], df['v_fusion'], label='FUSION', color='#e74c3c', linewidth=4)

    # Si tu as simulé une Ground Truth (target_v), on l'affiche en pointillés
    if 'target_v' in df.columns and not df['target_v'].isnull().all():
         ax_v.plot(df['temps'], df['target_v'], label='Vérité Terrain', color='black', linewidth=3, linestyle=':')

    ax_v.legend(loc='upper right')
    ax_v.grid(True, linestyle=':', alpha=0.6)

    # ==========================================
    # GRAPHIQUE 2 : AROUSAL
    # ==========================================
    ax_a.set_title("Évolution de l'Arousal", fontsize=14)
    ax_a.set_xlabel("Temps (secondes)", fontsize=12)
    ax_a.set_ylabel("Arousal (-1.0 à 1.0)", fontsize=12)
    ax_a.set_ylim(-1.05, 1.05)
    ax_a.axhline(0, color='black', linewidth=1, linestyle='--')

    ax_a.plot(df['temps'], df['a_vision'], label='Vision (ViT)', color='#3498db', linewidth=2, alpha=0.7)
    ax_a.plot(df['temps'], df['a_texte'], label='Texte (CamemBERT)', color='#2ecc71', linewidth=2, alpha=0.7)
    ax_a.plot(df['temps'], df['a_audio'], label='Audio/Ton (Wav2Vec2)', color='#f39c12', linewidth=2, alpha=0.7)
    
    ax_a.plot(df['temps'], df['a_fusion'], label='FUSION', color='#e74c3c', linewidth=4)

    if 'target_a' in df.columns and not df['target_a'].isnull().all():
         ax_a.plot(df['temps'], df['target_a'], label='Vérité Terrain', color='black', linewidth=3, linestyle=':')

    ax_a.legend(loc='upper right')
    ax_a.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()

    # 3. Sauvegarde automatique de l'image
    chemin_image = os.path.join(CSV_DIR, f"{nom_fichier}_graphique.png")
    plt.savefig(chemin_image, dpi=300, bbox_inches='tight')
    print(f"✅ Graphique sauvegardé en HD : {chemin_image}")

    # 4. Affichage interactif
    plt.show()

if __name__ == "__main__":
    fichiers_fused = glob.glob(os.path.join(CSV_DIR, "*_fused.csv"))
    
    if not fichiers_fused:
        print(f"❌ Aucun fichier _fused.csv trouvé dans {CSV_DIR}")
    else:
        # Boucle sur tous les fichiers fusionnés pour générer tous les graphiques d'un coup
        for fichier in fichiers_fused:
            tracer_graphiques(fichier)