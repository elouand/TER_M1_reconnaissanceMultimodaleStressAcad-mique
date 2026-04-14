import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

def tracer_graphiques(chemin_csv):
    if not os.path.exists(chemin_csv):
        print(f"Erreur : Le fichier {chemin_csv} n'existe pas.")
        return

    print(f"Chargement des données depuis {chemin_csv}...")
    
    # Lecture du CSV
    df = pd.read_csv(chemin_csv)

    df = df.sort_values(by='timestamp').reset_index(drop=True)
    
    # Création d'une colonne de temps relatif (commence à 0 seconde)
    df['temps_relatif'] = df['timestamp'] - df['timestamp'].iloc[0]

    # Paramétrage de la figure (2 graphiques l'un au-dessus de l'autre)
    fig, (ax_valence, ax_arousal) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Analyse Émotionnelle Multimodale\nFichier: {os.path.basename(chemin_csv)}", fontsize=16, fontweight='bold')

    # ==========================================
    # GRAPHIQUE 1 : VALENCE (Positif / Négatif)
    # ==========================================
    ax_valence.set_title("VALENCE ( -1 = Négatif | +1 = Positif )", fontsize=14)
    
    # Lignes des modalités (en pointillés et plus fines)
    ax_valence.plot(df['temps_relatif'], df['vv'], label='Vision (Sourire/Tristesse)', color='green', linestyle='--', alpha=0.7)
    ax_valence.plot(df['temps_relatif'], df['va'], label='Audio (Ton de la voix)', color='blue', linestyle='--', alpha=0.7)
    ax_valence.plot(df['temps_relatif'], df['vt'], label='Texte (Sens des mots)', color='orange', linestyle='--', alpha=0.7)
    
    # Ligne de FUSION (en gras)
    ax_valence.plot(df['temps_relatif'], df['v'], label='FUSION GLOBALE', color='black', linewidth=3)
    
    ax_valence.set_ylim(-1.1, 1.1)
    ax_valence.axhline(0, color='gray', linestyle='-', linewidth=1) # Ligne du zéro
    ax_valence.set_ylabel("Score de Valence")
    ax_valence.legend(loc='upper right')
    ax_valence.grid(True, linestyle=':', alpha=0.6)

    # ==========================================
    # GRAPHIQUE 2 : AROUSAL (Calme / Excité)
    # ==========================================
    ax_arousal.set_title("AROUSAL ( -1 = Calme/Fatigué | +1 = Actif/Stressé )", fontsize=14)
    
    # Lignes des modalités
    ax_arousal.plot(df['temps_relatif'], df['av'], label='Vision (Intensité du visage)', color='green', linestyle='--', alpha=0.7)
    ax_arousal.plot(df['temps_relatif'], df['aa'], label='Audio (Volume/Énergie)', color='blue', linestyle='--', alpha=0.7)
    ax_arousal.plot(df['temps_relatif'], df['at'], label='Texte (Mots forts)', color='orange', linestyle='--', alpha=0.7)
    
    # Ligne de FUSION
    ax_arousal.plot(df['temps_relatif'], df['a'], label='FUSION GLOBALE', color='black', linewidth=3)
    
    ax_arousal.set_ylim(-1.1, 1.1)
    ax_arousal.axhline(0, color='gray', linestyle='-', linewidth=1) # Ligne du zéro
    ax_arousal.set_xlabel("Temps (Secondes)")
    ax_arousal.set_ylabel("Score d'Arousal")
    ax_arousal.legend(loc='upper right')
    ax_arousal.grid(True, linestyle=':', alpha=0.6)

    # ==========================================
    # AJOUT DES MARQUEURS (Touche 'M')
    # ==========================================
    # On cherche les moments où mark == 1
    marques = df[df['mark'] == 1]
    for _, ligne in marques.iterrows():
        t_marque = ligne['temps_relatif']
        # Ligne verticale rouge sur les deux graphiques
        ax_valence.axvline(x=t_marque, color='red', linestyle=':', linewidth=2, label='Touche M' if _ == marques.index[0] else "")
        ax_arousal.axvline(x=t_marque, color='red', linestyle=':', linewidth=2)

    # Ajustement de la mise en page
    plt.tight_layout()
    plt.subplots_adjust(top=0.92) # Laisse un peu de place pour le titre global

    # Sauvegarde et affichage
    nom_image = chemin_csv.replace('.csv', '.png')
    plt.savefig(nom_image)
    print(f"Graphique sauvegardé sous : {nom_image}")
    
    plt.show()

if __name__ == "__main__":
    # Si on passe un fichier en argument dans le terminal : python visualiser_donnees.py csv/image_1/fichier.csv
    if len(sys.argv) > 1:
        fichier_cible = sys.argv[1]
    else:
        # Sinon, tu peux mettre le chemin en dur ici pour tester :
        fichier_cible = "csv/test_20260413_102702.csv" # <--- Modifie ce chemin avec le nom de ton vrai CSV
        
        print("💡 Astuce : Vous pouvez aussi lancer le script via le terminal :")
        print("python visualiser_donnees.py chemin/vers/le/fichier.csv\n")
    
    tracer_graphiques(fichier_cible)