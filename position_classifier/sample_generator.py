import polars as pl
import json
import os
import configparser
import gc
import matplotlib.pyplot as plt
from tqdm import tqdm

def plot_theme_distribution(final_df, theme_to_index, save_path):
    """Génère un bar plot avec échelle standard et valeurs par barre."""
    print("📊 Génération du graphique de distribution (Échelle standard + Valeurs)...")
    
    # 1. Préparation des données
    counts = (
        final_df.select(pl.col("Themes").explode())
        .group_by("Themes")
        .len()
    )
    if isinstance(counts, pl.LazyFrame): counts = counts.collect()

    plot_data = sorted([
        {"index": theme_to_index[row["Themes"]], "name": row["Themes"], "count": row["len"]}
        for row in counts.iter_rows(named=True) if row["Themes"] in theme_to_index
    ], key=lambda x: x["index"])

    indices = [d["index"] for d in plot_data]
    frequencies = [d["count"] for d in plot_data]
    names = [d["name"] for d in plot_data]

    # 2. Création du graphique
    plt.figure(figsize=(14, 7))
    bars = plt.bar(indices, frequencies, color='skyblue', edgecolor='navy', alpha=0.8)
    
    # --- AFFICHAGE DES VALEURS ---
    # On ajoute le texte au-dessus de chaque barre
    max_val = max(frequencies) if frequencies else 1
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + (max_val * 0.01), 
                 int(yval), va='bottom', ha='center', fontsize=9, fontweight='bold')

    # --- CONFIGURATION DES AXES ---
    plt.xlabel("Index du Thème", fontsize=11)
    plt.ylabel("Nombre de puzzles", fontsize=11)
    plt.title("Distribution des Thèmes dans le Sample", fontsize=14, pad=15)
    
    # Grille simple sur l'axe Y pour aider la lecture sans surcharger
    plt.grid(axis='y', linestyle='--', alpha=0.3)
    
    # Noms des thèmes en rotation
    plt.xticks(indices, [f"{i}: {n}" for i, n in zip(indices, names)], rotation=45, ha='right', fontsize=9)

    # Nettoyage des bordures
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    
    # --- SAUVEGARDE ---
    os.makedirs(save_path, exist_ok=True)
    full_save_path = os.path.join(save_path, "theme_distribution.png")
    plt.savefig(full_save_path, bbox_inches='tight', dpi=100)
    print(f"✅ Graphique sauvegardé : {full_save_path}")
    plt.close()


def generate_balanced_sample(n_target, max_moves,data_dir_path, min_threshold_ratio=0.1):
    """
    Génère un dataset équilibré en piquant dans les partitions par rareté.
    Exclut les thèmes sous un seuil de présence réelle (primaire + secondaire).
    """
    balanced_dir = os.path.join(data_dir_path, "balanced")
    min_threshold_pourcent = int(min_threshold_ratio * 100)
    is_full_mode = (n_target == -1)
    suffix = "FULL" if is_full_mode else n_target
    subdir = os.path.join(balanced_dir, f"S{suffix}_M{max_moves}_T{min_threshold_pourcent}")
    index_output_path = os.path.join(subdir, f"theme_to_index_S{suffix}_M{max_moves}_T{min_threshold_pourcent}.json")
    dataframe_output_path = os.path.join(subdir, f"dataset_S{suffix}_M{max_moves}_T{min_threshold_pourcent}.parquet")
    
    if os.path.exists(balanced_dir) and os.path.exists(subdir) and os.path.exists(index_output_path) and os.path.exists(dataframe_output_path):
        print("✅ Échantillon équilibré déjà existant.")
        return

    partitions_dir = os.path.join(data_dir_path, "raw/partitioned_themes")
    rarity_json = os.path.join(data_dir_path, "raw/theme_rarity_ranking.json")
    # --- 1. CHARGEMENT DES PARAMÈTRES ET DU RARETY RANKING ---
    if not os.path.exists(rarity_json):
        print(f"❌ Erreur : '{rarity_json}' introuvable. Lancez le script de partitionnement d'abord.")
        return

    with open(rarity_json, "r") as f:
        rarity_map = json.load(f)
    
    # Thèmes triés du plus rare au plus fréquent
    sorted_themes = sorted(rarity_map.keys(), key=lambda t: rarity_map[t])

    # --- 2. INITIALISATION DES POOLS (CHARGEMENT EN MÉMOIRE) ---
    pools = {}
    print(f"📥 Chargement des partitions et filtrage (n_moves < {max_moves})...")
    
    for theme in tqdm(sorted_themes, desc="Lecture des fichiers"):
        file_path = os.path.join(partitions_dir, f"{theme}.parquet")
        if os.path.exists(file_path):
            # On ne charge que les colonnes nécessaires pour économiser la RAM
            df = pl.read_parquet(file_path).filter(pl.col("n_moves") < max_moves)
            if len(df) > 0:
                # Stockage en liste de dicts pour un pop() ultra-rapide
                pools[theme] = df.to_dicts()

    if not pools:
        print("⚠️ Aucun puzzle ne correspond aux critères.")
        return

    picked_puzzles = []
    picked_ids = set()
    # theme_distribution suivra la présence RÉELLE (multi-label) dans le sample
    theme_distribution = {t: 0 for t in sorted_themes}

    # --- 3. CAS A : MODE COMPLET (FULL EXPORT) ---
    if is_full_mode:
        print(f"🚀 Mode COMPLET : Fusion de toutes les partitions valides...")
        for theme in tqdm(sorted_themes, desc="Fusion"):
            if theme in pools:
                for puzzle in pools[theme]:
                    picked_puzzles.append(puzzle)
        final_df = pl.from_dicts(picked_puzzles)

    # --- 4. CAS B : ÉCHANTILLONNAGE CYCLIQUE ÉQUILIBRÉ ---
    else:
        # --- PHASE 1 : CYCLE DE DÉCOUVERTE (PIOCHE INITIALE) ---
        n_themes_init = len(pools)
        theoretical_quota = n_target // n_themes_init
        min_count_required = int(theoretical_quota * min_threshold_ratio)
        
        print(f"🔄 Cycle 1 : Évaluation de la distribution (Seuil : {min_count_required} puzzles)")
        
        for theme in tqdm(sorted_themes, desc="Pioche stratégique"):
            if theme not in pools: continue
            
            # On pioche le quota théorique (si disponible)
            to_pick = min(len(pools[theme]), theoretical_quota)
            
            for _ in range(to_pick):
                puzzle = pools[theme].pop()
                if puzzle["PuzzleId"] not in picked_ids:
                    picked_puzzles.append(puzzle)
                    picked_ids.add(puzzle["PuzzleId"])
                    # On crédite TOUS les thèmes du puzzle dans la distribution
                    for t in puzzle["Themes"]:
                        if t in theme_distribution:
                            theme_distribution[t] += 1

        # --- PHASE 2 : FILTRAGE PAR LE SEUIL INTELLIGENT ---
        # On identifie les thèmes qui, même après les labels secondaires, restent trop rares
        themes_to_keep = [t for t, count in theme_distribution.items() if count >= min_count_required]
        themes_to_drop = [t for t in pools.keys() if t not in themes_to_keep]

        if themes_to_drop:
            print(f"\n🗑️ Exclusion de {len(themes_to_drop)} thèmes sous le seuil ({min_count_required} requis) :")
            # On trie par nombre d'occurrences pour y voir plus clair
            dropped_details = sorted(
                [(t, theme_distribution[t]) for t in themes_to_drop], 
                key=lambda x: x[1], 
                reverse=True
            )
            for theme_name, count in dropped_details:
                print(f"   - {theme_name}: {count} puzzles trouvés")
        else:
            print("\n✅ Aucun thème n'a été exclu par le seuil.")

        for t in themes_to_drop:
            if t in pools: del pools[t]

        # --- PHASE 3 : COMPLÉTION CYCLIQUE ---
        remaining_needed = n_target - len(picked_ids)
        if remaining_needed > 0 and pools:
            print(f"补 Complétion de {remaining_needed} puzzles avec les thèmes valides...")
            pbar = tqdm(total=remaining_needed, desc="Finalisation")
            
            while len(picked_ids) < n_target and pools:
                active_themes = list(pools.keys())
                for theme in active_themes:
                    if len(picked_ids) >= n_target: break
                    
                    if pools[theme]:
                        puzzle = pools[theme].pop()
                        if puzzle["PuzzleId"] not in picked_ids:
                            picked_puzzles.append(puzzle)
                            picked_ids.add(puzzle["PuzzleId"])
                            pbar.update(1)
                            # Update distribution
                            for t in puzzle["Themes"]:
                                if t in theme_distribution: theme_distribution[t] += 1
                    else:
                        del pools[theme]
            pbar.close()
        
        final_df = pl.from_dicts(picked_puzzles)

    # --- 5. NETTOYAGE FINAL ET EXPORT ---
    if len(final_df) == 0:
        print("❌ Résultat vide.")
        return

    # On ne garde que les thèmes réellement présents dans le DF final pour le JSON
    print("🧹 Nettoyage final du mapping des thèmes...")
    final_themes = (
        final_df.select(pl.col("Themes").explode())
        .unique().sort("Themes")["Themes"].to_list()
    )
    
    theme_to_index = {theme: idx for idx, theme in enumerate(final_themes)}
    
    #create sample directory if not exists
    os.makedirs(balanced_dir, exist_ok=True)
    os.makedirs(subdir, exist_ok=True)

    # Sauvegarde du JSON (Mapping définitif pour l'IA)
    with open(index_output_path, "w") as f:
        json.dump(theme_to_index, f, indent=4)

    # Sauvegarde du Parquet (Dataset prêt pour conversion HDF5)
    final_df.write_parquet(dataframe_output_path)

    plot_theme_distribution(final_df, theme_to_index, save_path=subdir)

    print(f"\n✨ OPÉRATION TERMINÉE ✨")
    print(f"📂 Dataset : {dataframe_output_path}")
    print(f"🏷️ Labels  : {len(theme_to_index)} thèmes dans {os.path.basename(index_output_path)}")
    print(f"📊 Total   : {len(final_df)} puzzles.")

def run_sampling_generation():
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.ini')
    config.read(config_path)
    
    # Lecture des paramètres
    n_target = config.getint('sampling', 'sample_size') # -1 pour tout avoir
    max_moves = config.getint('sampling', 'puzzle_len')
    data_dir_path = config.get('path', 'data_dir')
    threshold_percentage = config.getint('sampling', 'threshold_percentage')

    #parameters override for testing
    """
    n_target = 1000
    max_moves = 10
    threshold_percentage = 10
    """
    
    # Exécution
    generate_balanced_sample(
        n_target=n_target, 
        max_moves=max_moves, 
        data_dir_path=data_dir_path,
        min_threshold_ratio=threshold_percentage / 100.0
    )

# ========================================================================
# POINT D'ENTRÉE
# ========================================================================
if __name__ == '__main__':
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.ini')
    config.read(config_path)
    
    # Lecture des paramètres
    n_target = config.getint('sampling', 'sample_size') # -1 pour tout avoir
    max_moves = config.getint('sampling', 'puzzle_len')
    data_dir_path = config.get('path', 'data_dir')
    threshold_percentage = config.getint('sampling', 'threshold_percentage')

    #parameters override for testing
    """
    n_target = 1000
    max_moves = 10
    threshold_percentage = 10
    """
    
    # Exécution
    generate_balanced_sample(
        n_target=n_target, 
        max_moves=max_moves, 
        data_dir_path=data_dir_path,
        min_threshold_ratio=threshold_percentage / 100.0
    )