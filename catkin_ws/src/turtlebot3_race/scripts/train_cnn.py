import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv

class LaserDataset(Dataset):
    def __init__(self, csv_file):
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        self.X = data[:, :360].astype(np.float32)
        self.y = data[:, 360:362].astype(np.float32)  # Ajustado para tomar solo las dos primeras columnas de las etiquetas

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        scan = self.X[idx]
        scan = scan.reshape(1, 360)  # Para Conv1d: (canales, longitud)
        cmd = self.y[idx]
        return scan, cmd

class CNNController(nn.Module):
    def __init__(self):
        super(CNNController, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Linear(32 * 87, 128),  # Asegúrate de que este valor coincida con la salida de la capa convolucional
            nn.ReLU(),
            nn.Linear(128, 2)  # Salida ajustada a 2 para velocidad lineal y angular
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/dataset.csv')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', default='model.pt')
    args = parser.parse_args()

    dataset = LaserDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    model = CNNController()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs+1):
        total_loss = 0
        for scans, cmds in loader:
            optimizer.zero_grad()
            outputs = model(scans)
            loss = criterion(outputs, cmds)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch}: Loss={total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.save)
    print(f"Model saved to {args.save}")
