#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy.signal import medfilt
import argparse
import os

def is_spike(prev, curr, next_val, threshold=0.5):
    """Detecta picos bruscos con forma de valle repentino"""
    if prev <= 0 or curr <= 0 or next_val <= 0:
        return False
    drop1 = (prev - curr) / max(prev, 1e-5)
    rise = (next_val - curr) / max(curr, 1e-5)
    return drop1 > threshold and rise > threshold

def smooth_spikes(scan, threshold=0.5):
    """Corrige valores puntuales que bajan y suben bruscamente"""
    smoothed = scan.copy()
    for i in range(1, len(scan) - 1):
        if is_spike(scan[i-1], scan[i], scan[i+1], threshold):
            smoothed[i] = (scan[i-1] + scan[i+1]) / 2.0
    return smoothed

def main(input_csv, output_csv, max_range=10.0):
    df = pd.read_csv(input_csv)

    scan_cols = [col for col in df.columns if col.startswith('r')]
    scan_data = df[scan_cols].values
    other_data = df.iloc[:, 360:].values  # resto de columnas

    clean_rows = []
    prev_scan = None

    for i, scan in enumerate(scan_data):
        # Corregir valores inválidos y aplicar filtro de mediana
        scan = np.nan_to_num(scan, nan=max_range, posinf=max_range, neginf=0.0)
        scan = medfilt(scan, kernel_size=3)
        scan = smooth_spikes(scan, threshold=0.5)

        # Eliminar casi duplicados
        if prev_scan is not None:
            mse = np.mean((scan - prev_scan)**2)
            if mse < 1e-4:
                continue  # fila redundante
        prev_scan = scan.copy()

        row = list(scan) + list(other_data[i])
        clean_rows.append(row)

    # Guardar CSV filtrado
    header = scan_cols + df.columns[360:].tolist()
    filtered_df = pd.DataFrame(clean_rows, columns=header)
    filtered_df.to_csv(output_csv, index=False)
    print(f"Archivo filtrado guardado en: {output_csv}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Filtrar dataset de escaneos LiDAR")
    parser.add_argument('--input', type=str, required=True, help="Ruta al CSV original")
    parser.add_argument('--output', type=str, required=True, help="Ruta al CSV filtrado")
    args = parser.parse_args()

    main(args.input, args.output)
