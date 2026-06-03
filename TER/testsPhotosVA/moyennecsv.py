import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def _detect_delimiter(path):
    with open(path, newline='', encoding='utf-8') as f:
        sample = f.read(4096)
        if not sample:
            return ','
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        return dialect.delimiter

def calcul_moyenne(mode):
    if mode not in ("V", "A"):
        raise ValueError("Choix invalide, V ou A attendu.")

    f_name = "valence.csv" if mode == "V" else "arousal.csv"
    if not os.path.exists(f_name) or os.path.getsize(f_name) == 0:
        raise FileNotFoundError(f"{f_name} introuvable ou vide")

    delim = _detect_delimiter(f_name)
    with open(f_name, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter=delim)
        moyennes = {}
        for row in reader:
            for key, value in row.items():
                if key is None:
                    continue
                if key.strip().lower() in ('id_personne', 'id'):
                    continue
                if value is None or value.strip() == "":
                    continue
                try:
                    v = float(value)
                except ValueError:
                    continue
                moyennes.setdefault(key.strip(), []).append(v)

    if not moyennes:
        return {}

    resultats = {key: sum(vals) / len(vals) for key, vals in moyennes.items()}

    oasis_path = "oasis.csv"
    if not os.path.exists(oasis_path) or os.path.getsize(oasis_path) == 0:
        return {"resultats": resultats, "oasis": {}, "comparaison": {}}

    delim_oasis = _detect_delimiter(oasis_path)
    oasis_moyennes = {}
    key_col = "Valence_mean" if mode == "V" else "Arousal_mean"
    with open(oasis_path, mode='r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter=delim_oasis)
        for row in reader:
            image = row.get('Theme') or row.get('id_image')
            if not image:
                continue
            valeur = row.get(key_col)
            if valeur is None or valeur.strip() == "":
                continue
            try:
                oasis_moyennes[image.strip()] = float(valeur)
            except ValueError:
                continue

    comparaison = {}
    for image, m in resultats.items():
        if image in oasis_moyennes:
            comparaison[image] = {
                'moyenne_calculée': m,
                'oasis': oasis_moyennes[image],
                'différence': abs(m - oasis_moyennes[image])
            }

    return comparaison

def calcul_moyennes():
    cmp_v = calcul_moyenne("V")
    cmp_a = calcul_moyenne("A")

    keys = sorted(set(cmp_v.keys()) | set(cmp_a.keys()))
    merged = {}
    for k in keys:
        merged[k] = {
            "v_calc": cmp_v.get(k, {}).get("moyenne_calculée"),
            "a_calc": cmp_a.get(k, {}).get("moyenne_calculée"),
            "v_oasis": cmp_v.get(k, {}).get("oasis"),
            "a_oasis": cmp_a.get(k, {}).get("oasis"),
            "v_diff": cmp_v.get(k, {}).get("différence"),
            "a_diff": cmp_a.get(k, {}).get("différence"),
        }
    return merged

def plot_moyennes(data):
    labels = list(data.keys())
    n = len(labels)
    idx = np.arange(n)

    v_calc = [data[k]["v_calc"] if data[k]["v_calc"] is not None else np.nan for k in labels]
    a_calc = [data[k]["a_calc"] if data[k]["a_calc"] is not None else np.nan for k in labels]
    v_oasis = [data[k]["v_oasis"] if data[k]["v_oasis"] is not None else np.nan for k in labels]
    a_oasis = [data[k]["a_oasis"] if data[k]["a_oasis"] is not None else np.nan for k in labels]
    v_diff = [data[k]["v_diff"] if data[k]["v_diff"] is not None else np.nan for k in labels]
    a_diff = [data[k]["a_diff"] if data[k]["a_diff"] is not None else np.nan for k in labels]

    width = 0.35

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    ax = axes[0]
    ax.bar(idx - width/2, v_calc, width, label='V calculé', color='tab:blue', alpha=0.8)
    ax.bar(idx + width/2, a_calc, width, label='A calculé', color='tab:orange', alpha=0.8)
    ax.set_ylabel('moyenne')
    ax.set_title('Moyennes calculées (V/A)')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    ax = axes[1]
    ax.bar(idx - width/2, v_oasis, width, label='V oasis', color='tab:green', alpha=0.8)
    ax.bar(idx + width/2, a_oasis, width, label='A oasis', color='tab:red', alpha=0.8)
    ax.set_ylabel('moyenne')
    ax.set_title('Moyennes OASIS (V/A)')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    ax = axes[2]
    ax.bar(idx - width/2, v_diff, width, label='V différence', color='tab:purple', alpha=0.8)
    ax.bar(idx + width/2, a_diff, width, label='A différence', color='tab:brown', alpha=0.8)
    ax.set_ylabel('différence')
    ax.set_title('Différences absolues (V/A)')
    ax.legend()
    ax.grid(axis='y', linestyle=':', alpha=0.6)

    plt.xticks(idx, labels, rotation=45, ha='right')
    plt.xlabel('Image / thème')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    res = calcul_moyennes()
    for item, vals in res.items():
        fmt = lambda x: f"{x:.6f}" if isinstance(x, (int, float)) else "N/A"
        print(
            f"{item} : "
            f"({fmt(vals['v_calc'])}, {fmt(vals['a_calc'])}) | "
            f"({fmt(vals['v_oasis'])}, {fmt(vals['a_oasis'])}) | "
            f"({fmt(vals['v_diff'])}, {fmt(vals['a_diff'])})"
        )
    plot_moyennes(res)