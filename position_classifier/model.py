import torch
import torch.nn as nn
import configparser

# Lecture du fichier de configuration
config = configparser.ConfigParser()
config.read('position_classifier/config.ini')
# Récupération des paramètres du modèle
dropout_rate = config.getfloat('model', 'dropout_rate')
hidden_dim = config.getint('model', 'hidden_dim')

class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # On combine les 4 portes (input, forget, cell, output) en une seule convolution
        self.conv = nn.Conv2d(
            in_channels=self.input_dim + self.hidden_dim,
            out_channels=4 * self.hidden_dim,
            kernel_size=self.kernel_size,
            padding=self.padding,
            bias=self.bias
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        # Concaténation de l'entrée et de l'état caché précédent
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv(combined)

        # Séparation des 4 portes
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # Calcul du nouvel état de cellule et état caché
        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    
class ChessConvLSTMClassifier(nn.Module):
    def __init__(self, input_channels, hidden_dim, num_classes, dropout_rate, kernel_size=3):
        super(ChessConvLSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.cell = ConvLSTMCell(input_channels, hidden_dim, kernel_size)
        
        # Couche de sortie : Global Average Pooling + Linéaire
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # Réduit l'échiquier 8x8 à 1x1
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # x shape: (Batch, Time, Channels, Height, Width)
        # Exemple: (32, 5, 12, 8, 8) -> 32 puzzles, 5 coups, 12 types de pièces, 8x8
        
        batch_size, seq_len, _, h, w = x.size()
        
        # Initialisation des états cachés à zéro
        h_t = torch.zeros(batch_size, self.hidden_dim, h, w).to(x.device)
        c_t = torch.zeros(batch_size, self.hidden_dim, h, w).to(x.device)

        # On itère sur la séquence de coups
        for t in range(seq_len):
            h_t, c_t = self.cell(x[:, t, :, :, :], (h_t, c_t))

        # On classifie en utilisant le dernier état caché (après le dernier coup)
        return self.classifier(h_t)