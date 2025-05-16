import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os

class LaserDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        
        # Separar características y etiquetas
        self.X = data[:, :360].astype(np.float32)  # Datos del LiDAR
        self.y = data[:, 360:362].astype(np.float32)  # Velocidad lineal y angular (odom)
        
        # Aplicar normalización si se proporciona
        self.transform = transform
        if self.transform:
            self.X = self.transform(self.X)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        scan = self.X[idx]
        scan = scan.reshape(1, 360)  # Para Conv1d: (canales, longitud)
        cmd = self.y[idx]
        return scan, cmd

class CNNController(nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(CNNController, self).__init__()
        
        # Capas convolucionales con BatchNorm y Dropout
        self.conv = nn.Sequential(
            # Primera capa convolucional
            nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Segunda capa convolucional
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            # Tercera capa convolucional
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Calcular tamaño de salida de la capa convolucional para el input
        # Con los parámetros actuales, el tamaño será calculado automáticamente
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 360)
            dummy_output = self.conv(dummy_input)
            conv_output_size = dummy_output.size(1) * dummy_output.size(2)
        
        # Capas fully connected
        self.fc = nn.Sequential(
            nn.Linear(conv_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 2)  # Salida: velocidad lineal y angular
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--save', default='model.pt')
    parser.add_argument('--results_dir', default='results')
    args = parser.parse_args()
    
    # Crear directorio para resultados
    os.makedirs(args.results_dir, exist_ok=True)
    
    # Configurar dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Cargar y procesar datos
    print("Cargando datos...")
    
    # Normalizar datos
    data = np.loadtxt(args.data, delimiter=',', skiprows=1)
    scaler = StandardScaler()
    scaler.fit(data[:, :360])  # Ajustar solo con los datos de entrenamiento
    
    def transform_data(X):
        return scaler.transform(X)
    
    # Crear dataset
    dataset = LaserDataset(args.data, transform=transform_data)
    
    # Dividir en conjuntos de entrenamiento, validación y prueba
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch)
    
    # Crear modelo
    model = CNNController(dropout_rate=0.3).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)
    
    # Entrenar modelo
    print("Iniciando entrenamiento...")
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, device, args.epochs
    )
    
    # Guardar modelo
    torch.save({
        'model_state_dict': model.state_dict(),
        'scaler': scaler,
    }, args.save)
    print(f"Modelo guardado en {args.save}")
    
    # Evaluar modelo
    print("Evaluando modelo...")
    predictions, targets = evaluate_model(model, test_loader, criterion, device)
    
    # Visualizar resultados
    plot_losses(train_losses, val_losses, os.path.join(args.results_dir, 'loss_plot.png'))
    
    # Visualizar predicciones vs objetivos
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(targets[:, 0], predictions[:, 0], alpha=0.3)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('True Linear Velocity')
    plt.ylabel('Predicted Linear Velocity')
    plt.title('Linear Velocity Predictions')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(targets[:, 1], predictions[:, 1], alpha=0.3)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('True Angular Velocity')
    plt.ylabel('Predicted Angular Velocity')
    plt.title('Angular Velocity Predictions')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.results_dir, 'predictions_plot.png'))
    plt.show()