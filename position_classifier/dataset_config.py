import os
import gc
import h5py
import chess
import json
import configparser
import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset, DataLoader
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm

# ========================================================================
# 1. LOGIQUE TACTIQUE (LE "MOTEUR" DE TES TENSEURS)
# ========================================================================

_COORDS = np.array([[ (s % 8) / 7.0, (s // 8) / 7.0] for s in range(64)], dtype=np.float16)
VAL_ARRAY = np.array([0, 1, 3, 3, 5, 9, 100], dtype=np.float16)

def fast_create_position_tensor(board):
    """Calcule le tenseur 64x20 avec sécurité et influence tactique."""
    node_features = np.zeros((64, 10), dtype=np.float16)
    node_features[:, 7:9] = _COORDS
    piece_map = board.piece_map()
    sources, targets = [], []

    for square, piece in piece_map.items():
        p_type, p_color = piece.piece_type, piece.color
        node_features[square, p_type - 1] = 1.0
        node_features[square, 6] = 1.0 if p_color else -1.0
        
        # Sécurité
        atk_mask = board.attackers_mask(not p_color, square)
        if not atk_mask: safety = 1.0
        else:
            min_atk_val = 100
            for pt in range(1, 7):
                if atk_mask & board.pieces_mask(pt, not p_color):
                    min_atk_val = VAL_ARRAY[pt]
                    break
            def_mask = board.attackers_mask(p_color, square)
            is_my_turn = (board.turn == p_color)
            if def_mask == 0: safety = -1.0 if not is_my_turn else -0.5
            elif min_atk_val < VAL_ARRAY[p_type]: safety = -0.8 if not is_my_turn else -0.4
            elif atk_mask.bit_count() > def_mask.bit_count(): safety = -0.6 if not is_my_turn else -0.2
            else: safety = 0.4 if is_my_turn else 0.1
        node_features[square, 9] = safety

        move_mask = board.attacks_mask(square)
        targets.extend(list(chess.SquareSet(move_mask)))
        sources.extend([square] * move_mask.bit_count())

    nodes_t = torch.from_numpy(node_features)
    tactical_influence = torch.zeros((64, 10), dtype=torch.float16)
    if sources:
        tactical_influence.index_add_(0, torch.tensor(targets), nodes_t[sources])
    
    return torch.cat([nodes_t, tactical_influence], dim=-1).numpy()

def generate_sequence_tensors(fen, moves_string, max_t):
    """Génère la séquence temporelle avec padding."""
    board = chess.Board(fen)
    moves = moves_string.split() if moves_string else []
    sequence = np.full((max_t, 64, 20), -2.0, dtype=np.float16)
    
    sequence[0] = fast_create_position_tensor(board)
    for i, move_uci in enumerate(moves[:max_t-1]):
        try:
            board.push_uci(move_uci)
            sequence[i+1] = fast_create_position_tensor(board)
        except:
            sequence[i+1] = sequence[i]
    return sequence

# ========================================================================
# 2. CONVERSION (PARQUET TEXTE -> HDF5 TENSEURS)
# ========================================================================

def convert_to_hdf5_optimized(parquet_path, h5_path, theme_json, max_t=20, chunk_size=1000):
    if os.path.exists(h5_path):
        print(f"✅ HDF5 déjà présent : {h5_path}")
        return

    print(f"📦 Conversion de {parquet_path} en tenseurs tactiques...")
    df = pl.read_parquet(parquet_path)
    
    with open(theme_json, 'r') as f:
        theme_to_index = json.load(f)
    
    y_dim = len(theme_to_index)
    total_rows = len(df)

    with h5py.File(h5_path, 'w') as f:
        ds_x = f.create_dataset('features', shape=(total_rows, max_t, 64, 20), dtype='float32', compression="lzf")
        ds_y = f.create_dataset('labels', shape=(total_rows, y_dim), dtype='float32', compression="lzf")

        for i in tqdm(range(0, total_rows, chunk_size), desc="🔢 Processing"):
            batch = df.slice(i, chunk_size)
            curr_len = len(batch)

            # --- Traitement X (Génération des tenseurs) ---
            x_np = np.array([generate_sequence_tensors(r['FEN'], r['Moves'], max_t) 
                             for r in batch.iter_rows(named=True)], dtype=np.float32)

            # --- Traitement Y (One-hot encoding) ---
            y_np = np.zeros((curr_len, y_dim), dtype=np.float32)
            for idx, themes in enumerate(batch['Themes']):
                for t in themes:
                    if t in theme_to_index:
                        y_np[idx, theme_to_index[t]] = 1.0

            ds_x[i:i+curr_len] = x_np
            ds_y[i:i+curr_len] = y_np
            gc.collect()

# ========================================================================
# 3. PYTORCH DATASET & LOADERS
# ========================================================================

class ChessHDF5Dataset(Dataset):
    def __init__(self, h5_path, indices):
        self.h5_path = h5_path
        self.indices = indices
        self.db = None 

    def __len__(self): return len(self.indices)

    def __getitem__(self, idx):
        if self.db is None:
            self.db = h5py.File(self.h5_path, 'r', swmr=True)
        real_idx = self.indices[idx]
        return torch.from_numpy(self.db['features'][real_idx]), torch.from_numpy(self.db['labels'][real_idx])

def chess_collate_fn(batch):
    """ Reshape pour ConvLSTM : (B, T, 20, 8, 8) """
    xs, ys = zip(*batch)
    x_batch = torch.stack(xs)
    B, T, _, _ = x_batch.shape
    x_formatted = x_batch.view(B, T, 8, 8, 20).permute(0, 1, 4, 2, 3).contiguous()
    return x_formatted, torch.stack(ys)

# ========================================================================
# 4. FONCTION MAÎTRESSE : PREPARE & SPLIT
# ========================================================================

def prepare_data_and_loaders():
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.ini')
    config.read(config_path)

    data_dir_path = config.get('path', 'data_dir')
    sample_size = config.getint('sampling', 'sample_size')
    sample_threshold = config.getint('sampling', 'threshold_percentage')
    puzzle_len = config.getint('sampling', 'puzzle_len')
    chunk_size = config.getint('preprocessing', 'chunk_size')
    
    train_split = config.getfloat('preprocessing', 'train_split')
    val_split = config.getfloat('preprocessing', 'val_split')
    test_split = config.getfloat('preprocessing', 'test_split')
    batch_size = config.getint('preprocessing', 'batch_size')
    train_workers = config.getint('preprocessing', 'train_workers')
    val_workers = config.getint('preprocessing', 'val_workers')

    if sample_size == -1:
         # Cas où tu as fait un export FULL (à adapter selon ton nom de fichier)
         sample_size = "FULL" 
    sample_name = f"dataset_S{sample_size}_M{puzzle_len}_T{sample_threshold}.parquet"
    index_name = f"theme_to_index_S{sample_size}_M{puzzle_len}_T{sample_threshold}.json"
    sample_dir_path = os.path.join(data_dir_path, f"balanced/S{sample_size}_M{puzzle_len}_T{sample_threshold}")
    sample_path = os.path.join(sample_dir_path, sample_name)
    index_path = os.path.join(sample_dir_path, index_name)

    h5_path = os.path.join(data_dir_path, f"processed/chess_dataset_S{sample_size}_M{puzzle_len}_T{sample_threshold}.h5")
    
    # On vérifie que les fichiers du sampler existent
    if not os.path.exists(sample_path) or not os.path.exists(index_path):
        print("❌ Erreur : Fichiers du sampler introuvables. Lance d'abord le sampling.")
        return

    # 1. Conversion Tactique
    convert_to_hdf5_optimized(sample_path, h5_path, index_path, 
                               max_t=puzzle_len, chunk_size=chunk_size)

    # 2. Split Itératif
    print("⚖️ Calcul du split itératif...")
    with h5py.File(h5_path, 'r') as f:
        y_all = f['labels'][:] 

    nb_themes = y_all.shape[1]
    indices = np.arange(len(y_all)).reshape(-1, 1)
    
    idx_train, _, idx_temp, y_temp = iterative_train_test_split(indices, y_all, test_size=1-train_split)
    rel_test_size = test_split / (val_split + test_split)
    idx_val, _, idx_test, _ = iterative_train_test_split(idx_temp, y_temp, test_size=rel_test_size)
    
    del y_all, y_temp
    gc.collect()

    #vérifie le device disponible
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pin = False
    if device.type == 'cuda':
        pin = True

    # 3. DataLoaders
    train_loader = DataLoader(ChessHDF5Dataset(h5_path, idx_train.flatten()), batch_size=batch_size, shuffle=True, num_workers=train_workers, pin_memory=pin, collate_fn=chess_collate_fn)
    val_loader = DataLoader(ChessHDF5Dataset(h5_path, idx_val.flatten()), batch_size=batch_size, shuffle=False, num_workers=val_workers, pin_memory=pin, collate_fn=chess_collate_fn)
    test_loader = DataLoader(ChessHDF5Dataset(h5_path, idx_test.flatten()), batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=pin, collate_fn=chess_collate_fn)

    return train_loader, val_loader, test_loader,nb_themes

if __name__ == '__main__':
    t, v, te, nb_themes = prepare_data_and_loaders()
    x, y = next(iter(t))
    print(f"🚀 Batch OK ! Forme X : {x.shape} (Format ConvLSTM)")