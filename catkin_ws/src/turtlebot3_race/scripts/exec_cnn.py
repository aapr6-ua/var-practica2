#!/usr/bin/env python3
import rospy
import torch
import numpy as np
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
import os
import sys
from sklearn.preprocessing import StandardScaler

# Agregamos la implementación actualizada de CNNController directamente aquí para evitar
# problemas de importación con el módulo train_cnn

class CNNController(torch.nn.Module):
    def __init__(self, dropout_rate=0.2):
        super(CNNController, self).__init__()
        
        # Capas convolucionales con BatchNorm y Dropout
        self.conv = torch.nn.Sequential(
            # Primera capa convolucional
            torch.nn.Conv1d(1, 32, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm1d(32),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            
            # Segunda capa convolucional
            torch.nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm1d(64),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            
            # Tercera capa convolucional
            torch.nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm1d(128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
        )
        
        # Calcular tamaño de salida de la capa convolucional para el input
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 360)
            dummy_output = self.conv(dummy_input)
            conv_output_size = dummy_output.size(1) * dummy_output.size(2)
        
        # Capas fully connected
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(conv_output_size, 256),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(128, 2)  # Salida: velocidad lineal y angular
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

class CNNExecutor:
    def __init__(self):
        rospy.init_node('cnn_executor')
        
        # Cargar modelo y escalador
        model_path = rospy.get_param('~model_path', 'model.pt')
        rospy.loginfo(f"Cargando modelo desde: {model_path}")
        
        try:
            # Cargar el checkpoint que contiene el modelo y el escalador
            checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Crear e inicializar el modelo
            self.model = CNNController()
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            
            # Cargar el escalador
            self.scaler = checkpoint['scaler']
            
            rospy.loginfo("Modelo y escalador cargados correctamente")
        except Exception as e:
            rospy.logerr(f"Error al cargar el modelo: {str(e)}")
            raise e

        # Publisher y Subscriber
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        self.scan = None
        self.rate = rospy.Rate(10)

    def scan_cb(self, msg):
        # Convertir datos del LiDAR a numpy array
        arr = np.array(msg.ranges)
        
        # Reemplazar valores no numéricos
        arr = np.nan_to_num(arr, nan=10.0, posinf=10.0, neginf=0.0)
        
        # Normalizar datos usando el escalador entrenado
        arr_2d = arr.reshape(1, -1)  # Reshape para el escalador
        arr_normalized = self.scaler.transform(arr_2d)
        arr_normalized = arr_normalized.flatten()  # Volver a 1D
        
        # Convertir a tensor para el modelo
        self.scan = torch.tensor(arr_normalized, dtype=torch.float32).view(1, 1, -1)

    def run(self):
        rospy.loginfo("Iniciando la ejecución del modelo CNN...")
        while not rospy.is_shutdown():
            if self.scan is not None:
                try:
                    with torch.no_grad():
                        cmd = self.model(self.scan).squeeze().numpy()
                    
                    # Crear mensaje Twist con las velocidades predichas
                    twist = Twist()
                    twist.linear.x = float(cmd[0])
                    twist.angular.z = float(cmd[1])
                    
                    # Publicar el comando de velocidad
                    self.pub.publish(twist)
                    
                except Exception as e:
                    rospy.logerr(f"Error en inferencia del modelo: {str(e)}")
            
            self.rate.sleep()

if __name__ == '__main__':
    try:
        executor = CNNExecutor()
        executor.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Error general: {str(e)}")