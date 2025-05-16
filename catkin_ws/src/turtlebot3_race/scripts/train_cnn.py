import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import csv
from scipy.signal import medfilt

class LaserDataset(Dataset):
    def __init__(self, csv_file, max_range=5.0):
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        self.X = data[:, :360].astype(np.float32)
        self.y = data[:, 360:362].astype(np.float32)  # Ajustado para tomar solo las dos primeras columnas de las etiquetas

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        scan = self.X[idx]
        scan = self.preprocess_scan(scan)
        scan = scan.reshape(1, 360)
        cmd = self.y[idx]
        return scan, cmd

class TurtlebotCNN(nn.Module):
    def __init__(self):
        super(TurtlebotCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.MaxPool1d(2)
        self.fc1 = nn.Linear(64 * 45, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    for epoch in range(1, epochs+1):
        # Modo entrenamiento
        model.train()
        train_loss = 0.0
        
        for scans, cmds in train_loader:
            scans, cmds = scans.to(device), cmds.to(device)
            
            optimizer.zero_grad()
            outputs = model(scans)
            loss = criterion(outputs, cmds)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Modo evaluación
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for scans, cmds in val_loader:
                scans, cmds = scans.to(device), cmds.to(device)
                outputs = model(scans)
                loss = criterion(outputs, cmds)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Ajustar learning rate
        scheduler.step(val_loss)
        
        # Guardar el mejor modelo
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict().copy()
        
        print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    # Restaurar el mejor modelo
    model.load_state_dict(best_model_state)
    
    return model, train_losses, val_losses

def plot_losses(train_losses, val_losses, save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
    plt.show()

def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0.0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for scans, cmds in test_loader:
            scans, cmds = scans.to(device), cmds.to(device)
            outputs = model(scans)
            loss = criterion(outputs, cmds)
            test_loss += loss.item()
            
            predictions.append(outputs.cpu().numpy())
            targets.append(cmds.cpu().numpy())
    
    test_loss /= len(test_loader)
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    
    # Calcular métricas adicionales
    mse_lin = np.mean((predictions[:, 0] - targets[:, 0]) ** 2)
    mse_ang = np.mean((predictions[:, 1] - targets[:, 1]) ** 2)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"MSE Linear Velocity: {mse_lin:.4f}")
    print(f"MSE Angular Velocity: {mse_ang:.4f}")
    
    return predictions, targets

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/dataset.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save', default='model.pt')
    parser.add_argument('--results_dir', default='results')
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
