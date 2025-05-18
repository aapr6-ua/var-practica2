#!/usr/bin/env python3
import rospy
import torch
import numpy as np
from sensor_msgs.msg import LaserScan, Imu
from geometry_msgs.msg import Twist
import os
from scipy.signal import medfilt

class TurtlebotCNN(torch.nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(TurtlebotCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.conv3 = torch.nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.pool = torch.nn.MaxPool1d(2)
        self.fc1 = torch.nn.Linear(64 * 45, 128)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.fc2 = torch.nn.Linear(128, 64)
        self.fc3 = torch.nn.Linear(64, 2)

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

class CNNController:
    def __init__(self):
        rospy.init_node('cnn_controller')
        
        # Cargar parámetros
        model_path = rospy.get_param('~model_path', 'model.pt')
        self.max_range = rospy.get_param('~max_range', 10.0)
        self.max_lin_vel = rospy.get_param('~max_lin_vel', 0.5)  # m/s
        self.max_ang_vel = rospy.get_param('~max_ang_vel', 1.5)  # rad/s
        self.control_rate = rospy.get_param('~control_rate', 10)  # Hz
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Cargar modelo
        rospy.loginfo(f"Cargando modelo desde: {model_path}")
        self.model = self.load_model(model_path)
        
        # Publicadores y suscriptores
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.imu_pub = rospy.Publisher('/imu', Imu, queue_size=1)  # Nuevo: publicar en /imu
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # Variables de estado
        self.latest_scan = None
        self.running = True
        
        rospy.loginfo("CNN Controller inicializado")
        
    def load_model(self, model_path):
        """Cargar modelo desde archivo"""
        try:
            # Comprobar si existe el archivo
            if not os.path.exists(model_path):
                rospy.logerr(f"El archivo de modelo no existe: {model_path}")
                return None
                
            # Cargar modelo
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Inicializar modelo
            model = TurtlebotCNN().to(self.device)
            
            # Cargar estado del modelo
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()  # Establecer en modo evaluación
            rospy.loginfo(f"Modelo cargado correctamente en {self.device}")
            return model
        except Exception as e:
            rospy.logerr(f"Error al cargar el modelo: {e}")
            return None
    
    def scan_callback(self, msg):
        """Callback para los datos del escáner láser"""
        ranges = np.array(msg.ranges)
        
        # Procesar datos inválidos
        ranges[np.isinf(ranges)] = self.max_range
        ranges[np.isnan(ranges)] = self.max_range
        ranges[ranges == 0] = self.max_range
        
        # Limitar rango
        ranges = np.clip(ranges, 0.0, self.max_range)
        
        self.latest_scan = ranges
    
    def preprocess_scan(self, scan):
        """Preprocesamiento del escaneo láser para la CNN"""
        # Aplicar filtro de mediana para reducir ruido
        scan_filtered = medfilt(scan, kernel_size=3)
        
        # Normalizar al rango [0,1]
        scan_normalized = scan_filtered / self.max_range
        
        # Convertir a tensor y dar formato para CNN
        scan_tensor = torch.FloatTensor(scan_normalized).reshape(1, 1, -1)
        return scan_tensor.to(self.device)
    
    def predict_velocity(self, scan):
        """Predecir comandos de velocidad a partir del escaneo láser"""
        if self.model is None:
            rospy.logerr("El modelo no está cargado, no se pueden hacer predicciones")
            return 0.0, 0.0
            
        with torch.no_grad():
            scan_tensor = self.preprocess_scan(scan)
            output = self.model(scan_tensor)
            lin_vel, ang_vel = output[0].cpu().numpy()
            
        # Limitar velocidades
        lin_vel = np.clip(lin_vel, -self.max_lin_vel, self.max_lin_vel)
        ang_vel = np.clip(ang_vel, -self.max_ang_vel, self.max_ang_vel)
        
        return lin_vel, ang_vel
    
    def run(self):
        """Bucle principal de control"""
        rate = rospy.Rate(self.control_rate)
        
        rospy.loginfo("Iniciando control del robot...")
        
        while not rospy.is_shutdown() and self.running:
            if self.latest_scan is not None:
                # Predecir velocidades
                lin_vel, ang_vel = self.predict_velocity(self.latest_scan)
                
                # Publicar comando de velocidad
                cmd_msg = Twist()
                cmd_msg.linear.x = lin_vel
                cmd_msg.angular.z = ang_vel
                self.cmd_pub.publish(cmd_msg)
                
                # Publicar velocidad angular en /imu
                imu_msg = Imu()
                imu_msg.header.stamp = rospy.Time.now()
                imu_msg.header.frame_id = "base_link"
                
                # Configurar velocidad angular (solo en el eje Z para yaw)
                imu_msg.angular_velocity.x = 0.0
                imu_msg.angular_velocity.y = 0.0
                imu_msg.angular_velocity.z = ang_vel
                
                # Establecer covarianzas como matrices de identidad
                imu_msg.orientation_covariance = [0.01 if i == 0 or i == 4 or i == 8 else 0.0 for i in range(9)]
                imu_msg.angular_velocity_covariance = [0.01 if i == 0 or i == 4 or i == 8 else 0.0 for i in range(9)]
                imu_msg.linear_acceleration_covariance = [0.01 if i == 0 or i == 4 or i == 8 else 0.0 for i in range(9)]
                
                # Publicar mensaje de IMU
                self.imu_pub.publish(imu_msg)
                
                # Log ocasional (cada 50 iteraciones)
                if rospy.get_time() % 5 < 0.1:
                    rospy.loginfo(f"Velocidad: lin={lin_vel:.2f} m/s, ang={ang_vel:.2f} rad/s")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = CNNController()
        controller.run()
    except rospy.ROSInterruptException:
        pass