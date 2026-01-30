import configparser
import os
import sys
import glob
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

def train_full_model(model, train_loader, val_loader, test_loader):
    
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.ini')
    config.read(config_path)

    epochs = config.getint('train', 'epochs')
    learning_rate= config.getfloat('train', 'learning_rate')
    model_save_path = config.get('train', 'model_save_path')
    pos_weight = config.getfloat('train', 'pos_weight')
    batch_size = config.getint('preprocessing', 'batch_size')
    dropout_rate = config.getfloat('model', 'dropout_rate')
    hidden_dim = config.getint('model', 'hidden_dim')
    sample_size = config.getint('sampling', 'sample_size')
    sample_threshold = config.getint('sampling', 'threshold_percentage')
    puzzle_len = config.getint('sampling', 'puzzle_len')

    model_dir_name = f"S{sample_size}_M{puzzle_len}_T{sample_threshold}__E{epochs}_L{learning_rate}_W{pos_weight}_B{batch_size}_D{dropout_rate}_H{hidden_dim}".replace('.', 'p')
    model_save_path = os.path.join(model_save_path, model_dir_name)

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
                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # --- MÉTRIQUES & LOGGING ---
        all_preds = np.vstack(all_preds)
        all_labels = np.vstack(all_labels)
        
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
    
    torch.save(model.state_dict(), f"{model_save_path}/final_model.pth")
    sys.exit(0) 