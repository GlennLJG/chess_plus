import configparser
import polars as pl
import numpy as np
import torch
from skmultilearn.model_selection import iterative_train_test_split
import torch
import gc
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

#lecture du fichier de config
config = configparser.ConfigParser()
config.read('position_classifier/config.ini')
#récupération des paramètressys.exit(0) 
formatage = config.get('dataset', 'formatage_type')
batch_size = config.getint('dataset', 'batch_size')
train_split = config.getfloat('dataset', 'train_split')
val_split = config.getfloat('dataset', 'val_split')
test_split = config.getfloat('dataset', 'test_split')
random_seed = config.getint('dataset', 'random_seed')
data_path = config.get('dataset', 'data_path')
index_path = config.get('dataset', 'index_path')

data=pl.read_parquet(data_path)

#split train/val/test
def split_data_iterative(df, theme_col='theme_vector', train_size=train_split, val_size=val_split, test_size=test_split):
    print("⚖️ Préparation du split itératif...")
    
    # 1. On prépare Y (matrice binaire des labels)
    # On convertit la colonne theme_vector en un array NumPy 2D
    y = np.array(df[theme_col].to_list())
    
    # 2. On prépare X comme une simple colonne d'indices (0, 1, 2, ..., N)
    # On reshape en (N, 1) car la fonction attend une matrice pour X
    indices = np.arange(len(df)).reshape(-1, 1)

    # --- ÉTAPE 1 : Split Train / (Val + Test) ---
    # iterative_train_test_split attend (X, y, test_size)
    idx_train, y_train, idx_temp, y_temp = iterative_train_test_split(
        indices, y, test_size = 1 - train_size
    )

    # --- ÉTAPE 2 : Split Val / Test ---
    # On recalcule le ratio relatif pour le bloc restant
    relative_test_size = test_size / (val_size + test_size)
    idx_val, y_val, idx_test, y_test = iterative_train_test_split(
        idx_temp, y_temp, test_size = relative_test_size
    )

    # 3. On récupère les DataFrames Polars correspondants via les indices
    # .flatten() pour repasser de (N, 1) à (N,)
    train_df = df.filter(pl.int_range(0, pl.len()).is_in(idx_train.flatten()))
    val_df   = df.filter(pl.int_range(0, pl.len()).is_in(idx_val.flatten()))
    test_df  = df.filter(pl.int_range(0, pl.len()).is_in(idx_test.flatten()))

    print(f"✅ Split terminé : Train({len(train_df)}), Val({len(val_df)}), Test({len(test_df)})")
    return train_df, val_df, test_df

train_df, val_df, test_df = split_data_iterative(data)


class ChessDataset(Dataset):
    def __init__(self, df):
        # On convertit les colonnes en listes pour un accès rapide par index
        # sans transformer les données en gros tenseurs tout de suite
        self.puzzles = df['all_tensor'].to_list()
        self.labels = df['theme_vector'].to_list()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # Conversion en tenseur au moment de l'appel
        x = torch.as_tensor(self.puzzles[idx], dtype=torch.float32)
        y = torch.as_tensor(self.labels[idx], dtype=torch.float32)
        return x, y
    
def chess_collate_fn(batch):
    # batch est une liste de tuples (x, y)
    xs, ys = zip(*batch)
    
    # 1. Padding dynamique (uniquement sur la longueur max du batch actuel)
    x_padded = pad_sequence(xs, batch_first=True, padding_value=-2)
    y_tensors = torch.stack(ys)
    
    # 2. Formatage ConvLSTM (B, T, 8, 8, 20) -> (B, T, 20, 8, 8)
    # On évite d'utiliser .view() sur tout le dataset d'un coup
    B, T, _, _ = x_padded.shape
    x_formatted = x_padded.view(B, T, 8, 8, 20).permute(0, 1, 4, 2, 3).contiguous()
    
    return x_formatted, y_tensors

# --- INITIALISATION DES DATASETS ---
train_ds = ChessDataset(train_df)
val_ds   = ChessDataset(val_df)
test_ds  = ChessDataset(test_df)

# On peut maintenant supprimer les DataFrames pour libérer de la RAM
del data, train_df, val_df, test_df
gc.collect()

# --- DATALOADERS ---
train_loader = DataLoader(
    train_ds, 
    batch_size=batch_size, 
    shuffle=True, 
    collate_fn=chess_collate_fn
)

val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=chess_collate_fn)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=chess_collate_fn)

# Vérification
sample_x, sample_y = next(iter(train_loader))
print("Batch shape:", sample_x.shape) # Doit afficher (B, T, 20, 8, 8)




