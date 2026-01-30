import polars as pl
import chess
import configparser
import torch
import torch.nn as nn
import numpy as np
import json
import os

#lecture du fichier de config
config = configparser.ConfigParser()
script_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(script_dir, 'config.ini')
config.read(config_path)
#récupération des paramètres
raw_data_path = config.get('preprocessing', 'raw_data_path')
sample_size = config.getint('preprocessing', 'sample_size')

raw_data=pl.read_csv(raw_data_path)

#limiter aux 100 premières lignes
raw_data=raw_data.head(sample_size)

useless_col=["GameUrl","Rating","RatingDeviation","NbPlays","OpeningTags"]
raw_data = raw_data.drop(useless_col)
# Ajoute la colonne "n_moves" à chaque ligne
raw_data = raw_data.with_columns(
    pl.col("Moves").str.split(" ").list.len().alias("n_moves")
)

#filter to keep only positions with less than 14 moves
filtered_data = raw_data.filter(pl.col("n_moves") < 10)

useless_themes = ["bishopEndgame", "endgame", "enPassant", "knightEndgame", "long", "master", 
                  "masterVsMaster", "middlegame", "oneMove", "opening", "pawnEndgame", 
                  "queenEndgame", "queenRookEndgame", "rookEndgame", "short", "superGM", 
                  "veryLong", "mix", "playerGames", "puzzleDownloadInformation","mat","castling","promotion","underpromotion"]

#retirer les lignes où tous les thèmes sont dans useless_themes
filtered_data = (
    filtered_data
    .with_columns(pl.col("Themes").str.split(" ")) 
    .filter(
        pl.col("Themes").list.set_difference(useless_themes).list.len() > 0
    )
)

# Compte le nombre de positions par thème
theme_distribution = (
    filtered_data
    .explode("Themes")
    .group_by("Themes")
    .len()
    .sort("len", descending=True)
)

#fait uen liste des thèmes avec moins de 10 problèmes
themes_to_remove = [row['Themes'] for row in theme_distribution.iter_rows(named=True) if row['len'] < 10]
themes_to_remove = list(set(themes_to_remove) - set(useless_themes))
print("Thèmes à retirer (moins de 10 problèmes):", themes_to_remove)

#pour chaque ligne, retirer les thèmes avec moins de 10 problèmes ou useless
final_data = filtered_data.with_columns(
    pl.col("Themes").list.filter(~pl.element().is_in(themes_to_remove+useless_themes))
).filter(pl.col("Themes").list.len() > 0)


# 1. On récupère la liste unique des thèmes
all_themes = final_data.select(pl.col("Themes").explode()).unique().sort("Themes").to_series().to_list()
theme_to_index = {theme: idx for idx, theme in enumerate(all_themes)}
#afficher le nombre de thèmes restants
print(f"Nombre de thèmes restants après filtrage : {len(theme_to_index)}")

# 2. On transforme la colonne "Themes" en une matrice NumPy (One-Hot Encoding multi-label)
# On utilise une liste de listes temporaire pour la conversion vers NumPy
def create_label_matrix(series, theme_map):
    num_rows = len(series)
    num_labels = len(theme_map)
    # On crée une matrice de zéros avec NumPy
    matrix = np.zeros((num_rows, num_labels), dtype=np.int8)
    
    for i, themes in enumerate(series):
        for theme in themes:
            if theme in theme_map:
                matrix[i, theme_map[theme]] = 1
    return matrix

# Extraction de la colonne "Themes" vers NumPy
# 1. Générer la matrice NumPy (comme on l'a fait avant)
themes_series = final_data["Themes"].to_list()
y_matrix = create_label_matrix(themes_series, theme_to_index)

# 2. L'ajouter au DataFrame proprement
# On convertit la matrice NumPy en une liste de listes pour que Polars l'accepte
final_data = final_data.with_columns(
    pl.Series("theme_vector", y_matrix.tolist())
)


# --- CONSTANTES PRÉ-CALCULÉES ---
_COORDS = np.array([[ (s % 8) / 7.0, (s // 8) / 7.0] for s in range(64)], dtype=np.float16)
VAL_ARRAY = np.array([0, 1, 3, 3, 5, 9, 100], dtype=np.float16)

def fast_create_position_tensor(board):
    """Calcule le tenseur (64, 20) pour l'état actuel du board."""
    node_features = np.zeros((64, 10), dtype=np.float16)
    node_features[:, 7:9] = _COORDS
    
    piece_map = board.piece_map()
    sources, targets = [], []

    for square, piece in piece_map.items():
        p_type, p_color = piece.piece_type, piece.color
        node_features[square, p_type - 1] = 1.0
        node_features[square, 6] = 1.0 if p_color else -1.0
        
        # Sécurité (9) via Bitboards
        atk_mask = board.attackers_mask(not p_color, square)
        if not atk_mask:
            safety = 1.0
        else:
            min_atk_val = 100
            for pt in range(1, 7):
                if atk_mask & board.pieces_mask(pt, not p_color):
                    min_atk_val = VAL_ARRAY[pt]
                    break
            
            def_mask = board.attackers_mask(p_color, square)
            is_my_turn = (board.turn == p_color)
            if def_mask == 0:
                safety = -1.0 if not is_my_turn else -0.5
            elif min_atk_val < VAL_ARRAY[p_type]:
                safety = -0.8 if not is_my_turn else -0.4
            elif atk_mask.bit_count() > def_mask.bit_count():
                safety = -0.6 if not is_my_turn else -0.2
            else:
                safety = 0.4 if is_my_turn else 0.1
        node_features[square, 9] = safety

        # Edges
        move_mask = board.attacks_mask(square)
        targets.extend(list(chess.SquareSet(move_mask)))
        sources.extend([square] * move_mask.bit_count())

    # Influence Tactique
    nodes_t = torch.from_numpy(node_features)
    tactical_influence = torch.zeros((64, 10), dtype=torch.float16)
    if sources:
        tactical_influence.index_add_(0, torch.tensor(targets), nodes_t[sources])
    
    return torch.cat([nodes_t, tactical_influence], dim=-1).numpy()

def generate_game_tensors(fen, moves_string):
    """
    Transforme une partie complète directement en une liste de tenseurs.
    """
    board = chess.Board(fen)
    moves = moves_string.split() if moves_string else []
    
    # 1. Tenseur de la position initiale
    results = [fast_create_position_tensor(board).tolist()]
    
    # 2. Tenseurs pour chaque coup joué
    for move_uci in moves:
        try:
            board.push_uci(move_uci)
            results.append(fast_create_position_tensor(board).tolist())
        except ValueError:
            # En cas de coup invalide, on duplique la position précédente pour garder la longueur
            results.append(results[-1])
            
    return results

# --- APPLICATION POLARS ---
final_data = final_data.with_columns(
    all_tensor = pl.struct(["FEN", "Moves"]).map_elements(
        lambda x: generate_game_tensors(x["FEN"], x["Moves"]),
        return_dtype=pl.List(pl.List(pl.List(pl.Float16)))
    )
)

#sauvegarde le dataframe final
final_data.write_parquet(f"position_classifier/data/processed/position_classifier_data_{sample_size}.parquet")

print(final_data.columns)
#sauvegarde le mapping thème-index
# Ensure the directory exists before running this
with open(f"position_classifier/data/processed/theme_to_index_{sample_size}.json", "w") as f:
    json.dump(theme_to_index, f, indent=4) # Added indent for readability

print("✅ Prétraitement terminé et données sauvegardées sous position_classifier/data/processed/.")