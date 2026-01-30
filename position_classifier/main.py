import train
import sample_generator
import dataset_config
import model

if __name__ == "__main__":

    sample_generator.run_sampling_generation()

    # Récupération des DataLoaders depuis dataset_config
    train_loader, val_loader, test_loader,output_size = dataset_config.prepare_data_and_loaders()

    # Initialisation du modèle
    model_instance = model.ChessConvLSTMClassifier(input_channels=20, hidden_dim=model.hidden_dim, num_classes=output_size, dropout_rate=model.dropout_rate)

    # Entraînement du modèle
    train.train_full_model(model_instance, train_loader, val_loader, test_loader)