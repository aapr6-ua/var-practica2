import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import csv
from scipy.signal import medfilt

class LaserDataset(Dataset):
    def __init__(self, csv_file, max_range=5.0):
        data = np.loadtxt(csv_file, delimiter=',', skiprows=1)
        self.X = data[:, :360].astype(np.float32)
        self.y = data[:, 360:362].astype(np.float32)
        self.max_range = max_range

    def preprocess_scan(self, scan):
        scan[scan == 0] = self.max_range
        scan = np.clip(scan, 0, self.max_range)
        scan /= self.max_range
        scan = medfilt(scan, kernel_size=5)
        if np.random.rand() < 0.5:
            scan = scan[::-1]
        scan += np.random.normal(0, 0.01, scan.shape)
        scan = np.clip(scan, 0, 1)
        return scan

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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/dataset.csv')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--save', default='model.pt')
    args = parser.parse_args()

    dataset = LaserDataset(args.data)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TurtlebotCNN().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for scans, cmds in loader:
            scans, cmds = scans.to(device), cmds.to(device)
            optimizer.zero_grad()
            outputs = model(scans)
            loss = criterion(outputs, cmds)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step(total_loss / len(loader))
        print(f"Epoch {epoch+1}/{args.epochs} - Loss: {total_loss/len(loader):.4f}")

    torch.save(model.state_dict(), args.save)
    print(f"Model saved to {args.save}")
