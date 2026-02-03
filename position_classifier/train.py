import configparser
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score

def plot_training_results(history, model_save_path, config):
    """Génère un diagnostic complet des performances avec légende des paramètres."""
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(21, 6))

    # 1. Courbes de Loss
    ax1.plot(history['t_loss'], label='Train Loss', color='#1f77b4', lw=2)
    ax1.plot(history['v_loss'], label='Val Loss', color='#ff7f0e', lw=2, linestyle='--')
    ax1.set_title("Évolution de la BCE Loss")
    ax1.set_xlabel("Époques")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Accuracy (Strict vs Hamming)
    ax2.plot(history['v_acc'], label='Exact Match (Strict)', color='#2ca02c', lw=2)
    ax2.plot(history['v_ham'], label='Hamming (Par Label)', color='#d62728', lw=2)
    ax2.set_title("Précision (%)")
    ax2.set_xlabel("Époques")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. F1-Score Macro
    ax3.plot(history['v_f1'], label='F1 Macro', color='#9467bd', lw=2)
    ax3.set_title("Équilibre des Classes (F1 Macro)")
    ax3.set_xlabel("Époques")
    ax3.set_ylabel("Score (0-1)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Ajout de la légende des paramètres d'entraînement (plus d'infos du config)
    train_params = (
        f"epochs={config.getint('train', 'epochs')}, "
        f"lr={config.getfloat('train', 'learning_rate')}, "
        f"pos_weight={config.getfloat('train', 'pos_weight')}, "
        f"batch_size={config.getint('preprocessing', 'batch_size')}, "
        f"sample_size={config.getint('sampling', 'sample_size')}, "
        f"dropout={config.getfloat('model', 'dropout_rate')}, "
        f"hidden_dim={config.getint('model', 'hidden_dim')}"
    )
    fig.suptitle(f"Paramètres d'entraînement : {train_params}", fontsize=14, color='navy')

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(model_save_path, exist_ok=True)
    # Cherche les fichiers existants
    save_path = os.path.join(model_save_path, "training_report")
    plt.savefig(save_path)
    print(f"\n[INFO] Rapport graphique sauvegardé : {save_path}")
    plt.close(fig)

def save_training_history(history, model_save_path, config, test_metrics):
    """Sauvegarde l'historique d'entraînement dans un fichier Parquet avec Pandas."""
    
    # Préparation des paramètres (dictionnaire)
    training_parameters = {
        "sample_size" : config.getint('sampling', 'sample_size'),
        "puzzle_len" : config.getint('sampling', 'puzzle_len'),
        "threshold_percentage" : config.getint('sampling', 'threshold_percentage'),
        "batch_size" : config.getint('preprocessing', 'batch_size'),
        "epochs" : config.getint('train', 'epochs'),
        "lr" : config.getfloat('train', 'learning_rate'),
        "pos_weight" : config.getfloat('train', 'pos_weight'),
        "dropout" : config.getfloat('model', 'dropout_rate'),
        "hidden_dim" : config.getint('model', 'hidden_dim')
    }
    
    loss_delta = (np.array(history['t_loss']) - np.array(history['v_loss'])).tolist()
    
    # Création de la nouvelle ligne
    # On met les données dans une liste [data] pour créer une seule ligne
    new_data = {
        "training_parameters": [training_parameters],
        "loss_delta": [loss_delta],
        "val_loss": [history['v_loss']],
        "val_hamming": [history['v_ham']],
        "val_exact_match": [history['v_acc']],
        "val_f1_macro": [history['v_f1']],
        "test_metrics": [test_metrics]
    }
    new_row = pd.DataFrame(new_data)

    os.makedirs(model_save_path, exist_ok=True)
    dataframe_name = "training_history.parquet"
    dataframe_path = os.path.join(model_save_path, dataframe_name)

    # Chargement ou création du DataFrame
    if os.path.exists(dataframe_path):
        # On lit l'ancien historique
        df_existing = pd.read_parquet(dataframe_path)
        # On ajoute la nouvelle ligne
        df = pd.concat([df_existing, new_row], ignore_index=True)
    else:
        # Premier entraînement, le DataFrame est juste la nouvelle ligne
        df = new_row

    # Sauvegarde
    df.to_parquet(dataframe_path, index=False)
    print(f"[INFO] Historique d'entraînement sauvegardé (Pandas) : {dataframe_path}")

def plot_trainings_history_df(model_save_path, report_indice):

    def format_params(params):
        """Transforme le dictionnaire de paramètres en une chaîne lisible pour la légende."""
        if isinstance(params, dict):
            # On crée une chaîne du type "lr: 0.001, batch: 32"
            return ", ".join([f"{k}: {p}" for k, p in params.items()])
        return str(params)

    """Génère des graphiques avec l'intégralité des paramètres en légende."""
    dataframe_name = "training_history.parquet"
    dataframe_path = os.path.join(model_save_path, dataframe_name)

    if not os.path.exists(dataframe_path):
        print(f"[WARN] Fichier d'historique non trouvé : {dataframe_path}")
        return

    df = pd.read_parquet(dataframe_path).reset_index(drop=True)
    num_exps = len(df)
    
    if num_exps == 0:
        return

    # --- Palette de couleurs ---
    cmap = plt.colormaps.get_cmap('turbo')
    colors = [cmap(i / max(1, num_exps - 1)) for i in range(num_exps)]

    # Création de la figure (un peu plus haute pour laisser de la place à la légende)
    fig = plt.figure(figsize=(32, 10))
    axes = [fig.add_subplot(1, 6, i+1) for i in range(5)]
    axes.append(fig.add_subplot(1, 6, 6, projection='polar'))

    metrics = ['loss_delta', 'val_loss', 'val_hamming', 'val_exact_match', 'val_f1_macro']
    titles = ["Delta Loss", "Validation Loss", "Val Hamming (%)", "Val Exact Match (%)", "Val F1 Macro"]

    # --- Graphiques 1 à 5 : Courbes par Époque ---
    for i, col in enumerate(metrics):
        for idx, row in df.iterrows():
            values = row[col]
            # On formate les paramètres pour l'étiquette de la légende
            label_params = format_params(row['training_parameters'])
            label_text = f"Exp {idx}: {label_params}"
            
            if isinstance(values, (list, np.ndarray)):
                axes[i].plot(range(len(values)), values, 
                             color=colors[idx], linewidth=2, 
                             label=label_text, alpha=0.8)
            else:
                axes[i].scatter(0, values, color=colors[idx], label=label_text)

        axes[i].set_title(titles[i], fontweight='bold', fontsize=14)
        axes[i].set_xlabel("Époque")
        axes[i].grid(True, alpha=0.3)

    # --- Graphique 6 : Radar ---
    test_metrics_keys = list(df['test_metrics'].iloc[0].keys())
    num_vars = len(test_metrics_keys)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]

    for idx, row in df.iterrows():
        values = [row['test_metrics'].get(k, 0) for k in test_metrics_keys]
        values += values[:1]
        axes[5].plot(angles, values, color=colors[idx], linewidth=2, alpha=0.6)
        axes[5].fill(angles, values, alpha=0.05, color=colors[idx])
    
    axes[5].set_xticks(angles[:-1])
    axes[5].set_xticklabels(test_metrics_keys, fontsize=9)

    # --- Gestion de la Légende ---
    # On récupère les handles d'un seul graphique pour ne pas doubler la légende
    handles, labels = axes[0].get_legend_handles_labels()
    
    # On place la légende très bas car elle risque d'être longue
    fig.legend(handles, labels, loc='upper center', 
               bbox_to_anchor=(0.5, 0.05), # Positionnée sous les graphiques
               ncol=1, # Une seule colonne pour pouvoir lire les paramètres longs
               fontsize=9, frameon=True, shadow=True)

    fig.suptitle(f"Rapport d'Entraînement détaillé #{report_indice}", fontsize=22, y=1.02)
    
    # Ajustement de la mise en page
    plt.tight_layout(rect=[0, 0.1, 1, 0.95]) # Laisse de la place en bas (0.1) pour la légende
    
    save_path = os.path.join(model_save_path, f"training_history_report_{report_indice}.svg")
    plt.savefig(save_path, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"[INFO] Rapport des entrainements généré sous {save_path}.")


def train_full_model(model, train_loader, val_loader, test_loader):
    
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.ini')
    config.read(config_path)

    epochs = config.getint('train', 'epochs')
    learning_rate= config.getfloat('train', 'learning_rate')
    model_dir_path = config.get('train', 'model_save_path')
    pos_weight = config.getfloat('train', 'pos_weight')
    batch_size = config.getint('preprocessing', 'batch_size')
    dropout_rate = config.getfloat('model', 'dropout_rate')
    hidden_dim = config.getint('model', 'hidden_dim')
    sample_size = config.getint('sampling', 'sample_size')
    sample_threshold = config.getint('sampling', 'threshold_percentage')
    puzzle_len = config.getint('sampling', 'puzzle_len')

    model_dir_name = f"S{sample_size}_M{puzzle_len}_T{sample_threshold}__E{epochs}_L{learning_rate}_W{pos_weight}_B{batch_size}_D{dropout_rate}_H{hidden_dim}".replace('.', 'p')
    model_save_path = os.path.join(model_dir_path, model_dir_name)

    # --- INITIALISATION ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_cuda = device.type == 'cuda'
    model.to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight).to(device))
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)
    
    # On définit le scaler. Il sera inactif sur CPU (enabled=False)
    scaler = torch.amp.GradScaler('cuda', enabled=is_cuda)

    best_f1 = 0
    history = {'t_loss': [], 'v_loss': [], 'v_acc': [], 'v_ham': [], 'v_f1': []}
    early_stop_patience = 5
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"🚀 Entraînement sur : {device.type.upper()}")
    if not is_cuda:
        print("💡 Note : Utilisation du Float32 natif (Optimisé pour Intel i7 13th Gen)")

    for epoch in range(epochs):
        # --- PHASE D'ENTRAÎNEMENT ---
        model.train()
        train_loss = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True).float()
            
            optimizer.zero_grad(set_to_none=True)
            
            # Autocast n'est activé QUE si on a un GPU. 
            # Sur CPU, on reste en Float32 pour la vitesse.
            with torch.amp.autocast(device_type=device.type, enabled=is_cuda):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            
            # Scaler gère automatiquement CPU (simple backward) vs GPU (scaled backward)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # --- PHASE DE VALIDATION ---
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device).float()
                
                with torch.amp.autocast(device_type=device.type, enabled=is_cuda):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.append(preds)
                all_labels.append(labels)

        # --- MÉTRIQUES & LOGGING ---
        all_preds = torch.cat(all_preds).cpu().numpy()
        all_labels = torch.cat(all_labels).cpu().numpy()
        
        avg_val_loss = val_loss / len(val_loader)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        hamming = (all_preds == all_labels).mean() * 100
        exact_match = np.all(all_preds == all_labels, axis=1).mean() * 100

        history['t_loss'].append(avg_train_loss)
        history['v_loss'].append(avg_val_loss)
        history['v_f1'].append(f1_macro)
        history['v_ham'].append(hamming)
        history['v_acc'].append(exact_match)
        
        # Récupération de l'heure actuelle
        heure_actuelle = datetime.now().strftime("%H:%M:%S")

        # --- LE NOUVEAU PRINT ---
        print(f"[{heure_actuelle}] Epoch {epoch+1:03d}/{epochs} | "
              f"T-Loss: {avg_train_loss:.4f} | "
              f"V-Loss: {avg_val_loss:.4f} | "
              f"F1: {f1_macro:.3f} | "
              f"Hamming: {hamming:.2f}%")

        # --- SAUVEGARDE & EARLY STOPPING ---
        if f1_macro > best_f1:
            best_f1 = f1_macro
            os.makedirs(model_save_path, exist_ok=True)
            torch.save(model.state_dict(), f"{model_save_path}/best_model.pth")
        
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve >= early_stop_patience:
            print("⏹️ Early stopping")
            break

    print("🏁 Entraînement terminé.")

    # 1. Graphiques
    plot_training_results(history, model_save_path, config)

    # --- PHASE DE TEST FINAL ---
    print("\n" + "="*50)
    print("🏁 ÉVALUATION FINALE (Sur Best Model)")
    print("="*50)
    
    if os.path.exists(f"{model_save_path}/best_model.pth"):
        model.load_state_dict(torch.load(f"{model_save_path}/best_model.pth", weights_only=True))
    
    model.eval()
    test_preds, test_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device).float()
            with torch.amp.autocast(device_type=device.type, enabled=is_cuda):
                outputs = model(inputs)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            test_preds.append(preds.cpu().numpy())
            test_labels.append(labels.cpu().numpy())

    test_preds = np.vstack(test_preds)
    test_labels = np.vstack(test_labels)

    print(f"📊 Résultats Test :")
    print(f"   -> Loss Finale : {avg_val_loss:.4f}")
    print(f"   -> Exact Match : {np.all(test_preds == test_labels, axis=1).mean()*100:.2f}%")
    print(f"   -> F1-Score Macro : {f1_score(test_labels, test_preds, average='macro', zero_division=0):.4f}")

    test_metrics = {
        "loss": avg_val_loss,
        "exact_match": np.all(test_preds == test_labels, axis=1).mean()*100,
        "f1_macro": f1_score(test_labels, test_preds, average='macro', zero_division=0)
    }
    save_training_history(history, model_dir_path, config, test_metrics)
    
    torch.save(model.state_dict(), f"{model_save_path}/final_model.pth")
    