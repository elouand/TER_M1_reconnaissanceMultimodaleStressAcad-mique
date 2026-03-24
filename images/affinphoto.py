import csv
import os
import random

# Images sélectionnées
images_fixes = [
    "Dog 29", "Waterfall 1", "Galaxy 3",
    "Intensity 1", "Gun 1", "Gun 2",
    "Fence 6", "Doctor 6", "Swingset 1"
]

def obtenir_prochain_id():
    """Parcourt les deux fichiers pour trouver l'ID le plus haut."""
    max_id = 0
    for f_name in ['valence.csv', 'arousal.csv']:
        if os.path.exists(f_name) and os.path.getsize(f_name) > 0:
            with open(f_name, mode='r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        cid = int(row['id_personne'])
                        if cid > max_id: max_id = cid
                    except: continue
    return max_id + 1

def lancer_experience():
    # 1. Choix du mode
    mode = ""
    while mode not in ["V", "A"]:
        mode = input("Évaluation de la (V)alence ou de l'(A)rousal ? [V/A] : ").upper()
    
    file_name = "valence.csv" if mode == "V" else "arousal.csv"
    label = "Valence" if mode == "V" else "Arousal"

    # 2. Préparation des colonnes (Triées par nom d'image)
    images_triees = sorted(images_fixes)
    fieldnames = ['id_personne'] + images_triees

    # 3. ID et mélange pour la session
    id_actuel = obtenir_prochain_id()
    images_random = list(images_fixes)
    random.shuffle(images_random)

    print(f"\n--- SESSION ID : {id_actuel} | MODE : {label} ---")
    print(f"Les données seront triées par colonnes : {', '.join(images_triees)}\n")

    # 4. Collecte des notes dans un dictionnaire
    reponses = {'id_personne': id_actuel}
    
    for i, img in enumerate(images_random, 1):
        val = input(f"[{i}/9] Note de {label} pour {img} (1-9) : ")
        reponses[img] = val

    # 5. Écriture dans le CSV
    file_exists = os.path.exists(file_name) and os.path.getsize(file_name) > 0
    
    with open(file_name, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        # Si le fichier est nouveau, on écrit l'en-tête (les noms des images)
        if not file_exists:
            writer.writeheader()
        
        # On écrit la ligne complète (le dictionnaire contient id + notes)
        writer.writerow(reponses)
        f.flush()

    print(f"\n--- TERMINÉ ---")
    print(f"Ligne ajoutée dans {file_name}. Voici le contenu actuel :\n")
    with open(file_name, 'r') as check_f:
        print(check_f.read())

if __name__ == "__main__":
    lancer_experience()