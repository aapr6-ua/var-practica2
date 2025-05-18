import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split, ConcatDataset
import numpy as np
import csv
import os
import glob
import matplotlib.pyplot as plt
from scipy.signal import medfilt
import argparse
from datetime import datetime

class LaserDataset(Dataset):
    def __init__(self, csv_file, max_range=10.0):
        """
        Dataset para datos de escaneo láser y comandos de velocidad
        
        Args:
            csv_file: Ruta al archivo CSV con los datos
            max_range: Rango máximo del láser para normalización
        """
        print(f"Cargando datos desde: {csv_file}")
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        print(f"Dataset cargado: {data.shape[0]} muestras")
        
        # Entradas: datos del láser (primeras 360 columnas)
        self.X = data[:, :360].astype(np.float32)
        
        # Salidas: velocidades de comando (columnas 362-363)
        # Índices 360:362 son velocidades de odometría (las ignoramos)
        self.y = data[:, 362:364].astype(np.float32)
        
        self.max_range = max_range
        print(f"Rango de valores X: [{np.min(self.X)}, {np.max(self.X)}]")
        print(f"Rango de valores y: [{np.min(self.y)}, {np.max(self.y)}]")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        scan = self.X[idx]
        scan = self.preprocess_scan(scan)
        scan = scan.reshape(1, 360)  # Formato para CNN (canales, longitud)
        cmd = self.y[idx]
        return scan, cmd
    
    def preprocess_scan(self, scan):
        """
        Preprocesamiento del escaneo láser
        
        Args:
            scan: Array de distancias del láser
            
        Returns:
            Array procesado y normalizado
        """
        # Aplicar filtro de mediana para eliminar ruido
        scan_filtered = medfilt(scan, kernel_size=3)
        
        # Normalizar al rango [0,1]
        scan_normalized = scan_filtered / self.max_range
        
        return scan_normalized

def load_all_datasets(data_dir, max_range=10.0):
    """
    Carga todos los datasets que comienzan con 'dataset_' en la carpeta de datos
    
    Args:
        data_dir: Ruta a la carpeta que contiene los datasets
        max_range: Rango máximo del láser para normalización
        
    Returns:
        Un ConcatDataset que contiene todos los datasets combinados
    """
    # Buscar todos los archivos que comienzan con 'dataset_' en la carpeta especificada
    dataset_pattern = os.path.join(data_dir, "dataset_*.csv")
    dataset_files = glob.glob(dataset_pattern)
    
    if not dataset_files:
        raise ValueError(f"No se encontraron archivos 'dataset_*.csv' en {data_dir}")
    
    print(f"Se encontraron {len(dataset_files)} archivos de dataset:")
    for file in dataset_files:
        print(f"  - {os.path.basename(file)}")
    
    # Cargar cada dataset y añadirlo a la lista
    datasets = []
    total_samples = 0
    
    for file_path in dataset_files:
        try:
            dataset = LaserDataset(file_path, max_range)
            datasets.append(dataset)
            total_samples += len(dataset)
        except Exception as e:
            print(f"Error al cargar {file_path}: {e}")
    
    if not datasets:
        raise ValueError("No se pudo cargar ningún dataset correctamente")
    
    # Combinar todos los datasets
    combined_dataset = ConcatDataset(datasets)
    print(f"Dataset combinado: {total_samples} muestras totales")
    
    return combined_dataset

class TurtlebotCNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        """
        CNN para el control del Turtlebot basado en datos de láser
        
        Args:
            dropout_rate: Tasa de dropout para regularización
        """
        super(TurtlebotCNN, self).__init__()
        
        # Capas convolucionales
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = nn.BatchNorm1d(64)
        
        # Pooling
        self.pool = nn.MaxPool1d(2)
        
        # Capas fully connected
        self.fc1 = nn.Linear(64 * 45, 128)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 2)  # Salida: [velocidad_lineal, velocidad_angular]

    def forward(self, x):
        # Capas convolucionales con activación ReLU y batch normalization
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = torch.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        
        # Aplanar para capas fully connected
        x = x.view(x.size(0), -1)
        
        # Capas fully connected con dropout
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        
        # Capa de salida (sin activación para regresión)
        return self.fc3(x)

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    """
    Entrenamiento del modelo con seguimiento de pérdidas
    
    Args:
        model: Modelo PyTorch a entrenar
        train_loader: DataLoader para datos de entrenamiento
        val_loader: DataLoader para datos de validación
        criterion: Función de pérdida
        optimizer: Optimizador
        scheduler: Scheduler para ajuste de learning rate
        device: Dispositivo (cuda/cpu)
        epochs: Número de épocas
        
    Returns:
        Modelo entrenado, historial de pérdidas
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None
    
    print(f"Iniciando entrenamiento en {device}...")
    
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
            print(f"Epoch {epoch}/{epochs}: Nuevo mejor modelo guardado (val_loss={val_loss:.4f})")
        
        print(f"Epoch {epoch}/{epochs}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
    
    # Restaurar el mejor modelo
    model.load_state_dict(best_model_state)
    print(f"Entrenamiento finalizado. Mejor val_loss: {best_val_loss:.4f}")
    
    return model, train_losses, val_losses

def plot_losses(train_losses, val_losses, save_path=None):
    """
    Visualización de pérdidas de entrenamiento y validación
    
    Args:
        train_losses: Lista de pérdidas de entrenamiento
        val_losses: Lista de pérdidas de validación
        save_path: Ruta para guardar el gráfico
    """
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
        print(f"Gráfico guardado en: {save_path}")
    plt.show()

def evaluate_model(model, test_loader, criterion, device):
    """
    Evaluación del modelo en conjunto de prueba
    
    Args:
        model: Modelo PyTorch a evaluar
        test_loader: DataLoader para datos de prueba
        criterion: Función de pérdida
        device: Dispositivo (cuda/cpu)
        
    Returns:
        Predicciones y valores reales
    """
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

def plot_predictions(predictions, targets, save_path=None):
    """
    Visualización de predicciones vs valores reales
    
    Args:
        predictions: Predicciones del modelo
        targets: Valores reales
        save_path: Ruta para guardar el gráfico
    """
    plt.figure(figsize=(12, 5))
    
    # Velocidad lineal
    plt.subplot(1, 2, 1)
    plt.scatter(targets[:, 0], predictions[:, 0], alpha=0.3)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('Target Linear Velocity')
    plt.ylabel('Predicted Linear Velocity')
    plt.title('Linear Velocity Predictions')
    plt.grid(True)
    
    # Velocidad angular
    plt.subplot(1, 2, 2)
    plt.scatter(targets[:, 1], predictions[:, 1], alpha=0.3)
    plt.plot([-1, 1], [-1, 1], 'r--')
    plt.xlabel('Target Angular Velocity')
    plt.ylabel('Predicted Angular Velocity')
    plt.title('Angular Velocity Predictions')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"Gráfico guardado en: {save_path}")
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Entrenar CNN para control de Turtlebot')
    parser.add_argument('--data_dir', type=str, default='data', help='Directorio que contiene los archivos CSV de datos')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas de entrenamiento')
    parser.add_argument('--batch', type=int, default=64, help='Tamaño de batch')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate inicial')
    parser.add_argument('--save', type=str, default='model.pt', help='Ruta para guardar el modelo')
    parser.add_argument('--results_dir', type=str, default='results', help='Directorio para resultados')
    args = parser.parse_args()
    
    # Crear directorio para resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = f"{args.results_dir}_{timestamp}"
    os.makedirs(results_dir, exist_ok=True)
    
    # Configurar dispositivo
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Usando dispositivo: {device}")
    
    # Cargar todos los datasets
    try:
        dataset = load_all_datasets(args.data_dir)
    except ValueError as e:
        print(f"Error: {e}")
        exit(1)
    
    # Dividir en conjuntos de entrenamiento, validación y prueba
    train_size = int(0.7 * len(dataset))
    val_size = int(0.15 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    
    print(f"Dataset dividido: {train_size} entrenamiento, {val_size} validación, {test_size} prueba")
    
    # Crear dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch)
    test_loader = DataLoader(test_dataset, batch_size=args.batch)
    
    # Crear modelo
    model = TurtlebotCNN().to(device)
    print(model)
    
    # Configurar entrenamiento
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    
    # Entrenar modelo
    model, train_losses, val_losses = train_model(
        model, train_loader, val_loader, criterion, 
        optimizer, scheduler, device, args.epochs
    )
    
    # Guardar modelo
    model_path = os.path.join(results_dir, args.save)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_losses': train_losses,
        'val_losses': val_losses,
    }, model_path)
    print(f"Modelo guardado en: {model_path}")
    
    # Visualizar pérdidas
    loss_plot_path = os.path.join(results_dir, 'loss_plot.png')
    plot_losses(train_losses, val_losses, loss_plot_path)
    
    # Evaluar modelo
    predictions, targets = evaluate_model(model, test_loader, criterion, device)
    
    # Visualizar predicciones
    pred_plot_path = os.path.join(results_dir, 'predictions_plot.png')
    plot_predictions(predictions, targets, pred_plot_path)
    
    print(f"Entrenamiento completado. Resultados guardados en: {results_dir}")