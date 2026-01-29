import train 
import dataset_config
import model
import json

if __name__ == "__main__":

    # Récupération des DataLoaders depuis dataset_config
    train_loader = dataset_config.train_loader
    val_loader = dataset_config.val_loader
    test_loader = dataset_config.test_loader

    # Initialisation du modèle
    # Récupération du mapping thème-index depuis le fichier JSON
    with open(dataset_config.index_path, "r") as f:
        theme_to_index = json.load(f)
    output_size = len(theme_to_index)  # Nombre de classes basé sur les thèmes uniques
    model = model.ChessConvLSTMClassifier(input_channels=20, hidden_dim=model.hidden_dim, num_classes=output_size, dropout_rate=model.dropout_rate)

    # Entraînement du modèle
    train.train_full_model(model, train_loader, val_loader, test_loader, train.epochs, train.learning_rate, train.model_save_path, train.loss_plot_path, train.pos_weight)