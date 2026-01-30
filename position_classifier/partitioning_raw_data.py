import polars as pl
import configparser
import os
import json
from tqdm import tqdm

def run_partitioning():
    # --- 1. CONFIGURATION ---
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.ini')
    config.read(config_path)
    
    data_dir_path = config.get('path', 'data_dir')
    output_dir = os.path.join(data_dir_path, "raw/partitioned_themes")
    os.makedirs(output_dir, exist_ok=True)

    useless_themes = ["bishopEndgame", "endgame", "enPassant", "knightEndgame", "long", "master", 
                      "masterVsMaster", "middlegame", "oneMove", "opening", "pawnEndgame", 
                      "queenEndgame", "queenRookEndgame", "rookEndgame", "short", "superGM", 
                      "veryLong", "mix", "playerGames", "puzzleDownloadInformation","crushing",
                      "mate","castling","promotion","underpromotion","advantage","defensiveMove","equality"]

    print("📖 Scan du dataset original...")
    lazy_df = (
        pl.scan_csv(os.path.join(data_dir_path, "raw/lichess_db_puzzle.csv"))
        .select(["PuzzleId", "FEN", "Moves", "Themes"])
        .with_columns(pl.col("Themes").str.split(" "))
        # On nettoie les thèmes inutiles tout de suite
        .with_columns(pl.col("Themes").list.filter(~pl.element().is_in(useless_themes)))
        .filter(pl.col("Themes").list.len() > 0)
    )

    # ajoute une colonne avec le nombre de coups par puzzle
    lazy_df = lazy_df.with_columns(
        pl.col("Moves").str.split(" ").list.len().alias("n_moves")
    )

    # --- 2. CALCUL DE LA RARETÉ GLOBALE ---
    print("📊 Calcul des fréquences de thèmes...")
    theme_counts = (
        lazy_df.select(pl.col("Themes").explode())
        .group_by("Themes")
        .len()
        .collect()
        .sort("len") # Du plus rare au plus fréquent
    )
    
    # On crée un dictionnaire de rareté (Theme -> Rang de rareté)
    # Plus le rang est bas, plus le thème est rare
    rarity_map = {row["Themes"]: i for i, row in enumerate(theme_counts.iter_rows(named=True))}
    
    with open(os.path.join(data_dir_path, "raw/theme_rarity_ranking.json"), "w") as f:
        json.dump(rarity_map, f, indent=4)

    # --- 3. ATTRIBUTION DU THÈME LE PLUS RARE PAR PUZZLE ---
    print("⚖️ Attribution du thème prioritaire (le plus rare)...")
    
    # On définit une fonction pour trouver le thème le plus rare dans une liste
    def get_rarest_theme(themes):
        # On cherche le thème avec l'index le plus bas dans rarity_map
        return min(themes, key=lambda t: rarity_map.get(t, 999))

    # On applique la logique
    # Note : On collect() ici car le traitement par ligne est nécessaire
    final_df = (
        lazy_df.collect()
        .with_columns(
            pl.col("Themes").map_elements(get_rarest_theme, return_dtype=pl.Utf8).alias("primary_theme")
        )
    )

    # --- 4. SAUVEGARDE PARTITIONNÉE ---
    print(f"💾 Sauvegarde des partitions dans {output_dir}...")
    
    # On utilise la puissance de Polars pour écrire un fichier par thème
    # Cela permet de charger uniquement les thèmes dont on a besoin plus tard
    unique_primary_themes = final_df["primary_theme"].unique().to_list()
    
    for theme in tqdm(unique_primary_themes, desc="Écriture des partitions"):
        theme_df = final_df.filter(pl.col("primary_theme") == theme)
        # On sauvegarde en Parquet (très compressé et rapide)
        theme_df.write_parquet(os.path.join(output_dir, f"{theme}.parquet"))

    print(f"✨ Partitionnement terminé. {len(unique_primary_themes)} thèmes créés.")

if __name__ == '__main__':
    run_partitioning()