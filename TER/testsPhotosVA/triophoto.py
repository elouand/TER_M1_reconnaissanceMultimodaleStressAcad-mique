import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def trouver_trios_proches(file_path):
    # 1. Lecture robuste : on laisse pandas deviner le séparateur (sep=None)
    # et on ignore les espaces autour des noms de colonnes (skipinitialspace)
    try:
        df = pd.read_csv(file_path, sep=None, engine='python', skipinitialspace=True)
    except Exception as e:
        return f"Erreur lors de la lecture du fichier : {e}"

    # Nettoyage de sécurité : on enlève les espaces invisibles dans les noms de colonnes
    df.columns = df.columns.str.strip()

    # Vérification de la présence des colonnes nécessaires
    cols_necessaires = ['Valence_mean', 'Arousal_mean', 'Theme']
    for col in cols_necessaires:
        if col not in df.columns:
            # Si la colonne n'est pas trouvée, on affiche les colonnes réelles pour aider au débug
            return f"Erreur : Colonne '{col}' manquante. Colonnes détectées : {list(df.columns)}"

    # 2. Préparation des données pour le clustering
    X = df[['Valence_mean', 'Arousal_mean']]
    
    # 3. K-Means pour trouver 3 zones éloignées
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10).fit(X)
    centroids = kmeans.cluster_centers_
    
    resultats = []
    
    # 4. Pour chaque centre, on trouve les 3 photos les plus proches
    for center in centroids:
        # Calcul de la distance euclidienne
        distances = np.sqrt(((X - center)**2).sum(axis=1))
        
        # On prend les 3 plus proches
        indices_proches = distances.nsmallest(3).index
        trio = df.loc[indices_proches, ['Theme', 'Valence_mean', 'Arousal_mean']]
        resultats.append(trio)
        
    return resultats

# --- EXÉCUTION ---
nom_fichier = 'oasis.csv' # Vérifiez bien que le nom est exact (majuscules/minuscules)
trios = trouver_trios_proches(nom_fichier)

if isinstance(trios, str):
    print(trios)
else:
    for idx, trio in enumerate(trios, 1):
        print(f"--- TRIO {idx} (Valeurs Proches) ---")
        print(trio.to_string(index=False))
        print("\n")