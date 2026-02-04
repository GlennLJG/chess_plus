import train
import sample_generator
import dataset_config
import model
import configparser
import os
import shutil
from sklearn.model_selection import ParameterGrid


if __name__ == "__main__":

    param_grid = {
        "sample_size": [10000],
        "learning_rates": [0.001,0.0005,0.0001],
        "batch_sizes": [256],
        "hidden_dims": [128],
        "puzzle_max_len": [10],
        "pos_weight": [3.0,8.0],
        "dropout_rates": [0.2]
    }
    
    config = configparser.ConfigParser()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config.ini')

    # Sauvegarde physique
    shutil.copy(config_path, os.path.join(script_dir, 'config.ini.bak'))

    try:
        #crée un dictionnaire des paramètres du fichier config.ini pour lui associer sa section
        config.read(config_path)
        param_to_section = {
            option: section 
            for section in config.sections() 
            for option in config.options(section)
        }

        for i, params in enumerate(ParameterGrid(param_grid)):
            print(f"\n[INFO] Grid search n° {i+1}/{len(list(ParameterGrid(param_grid)))+1} : {params}")
            # Lecture du fichier config.ini
            config = configparser.ConfigParser()
            config.read(config_path)
            # Mise à jour des paramètres dans config.ini
            for param, value in params.items():
                section = param_to_section.get(param)
                if section:
                    config.set(section, param, str(value))
            with open(config_path, 'w') as configfile:
                config.write(configfile)

            print(f"\n[INFO] constitution des données")
            sample_generator.run_sampling_generation()
            # Récupération des DataLoaders depuis dataset_config
            train_loader, val_loader, test_loader,output_size = dataset_config.prepare_data_and_loaders()

            print(f"[INFO] Entraînement du modèle")
            # Initialisation du modèle
            model_instance = model.ChessConvLSTMClassifier(input_channels=20, hidden_dim=model.hidden_dim, num_classes=output_size, dropout_rate=model.dropout_rate)
            # Entraînement du modèle
            train.train_full_model(model_instance, train_loader, val_loader, test_loader,config=config)
        
    finally:
        print("\n[INFO] Fin du grid search.")
        train.plot_trainings_history_df(model_save_path=config.get('train', 'model_save_path'), report_indice=0)

        # Restauration à la fin (qu'il y ait eu une erreur ou non)
        shutil.move(os.path.join(script_dir, 'config.ini.bak'), config_path)
        print("Fichier original restauré.")

    

    