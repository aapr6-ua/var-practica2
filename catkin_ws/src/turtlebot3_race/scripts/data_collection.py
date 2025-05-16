#!/usr/bin/env python3
import rospy
import csv
import os
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from datetime import datetime

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector')

        # Directorio de salida y archivo
        self.output_dir = rospy.get_param('~output_dir', 'data')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Usar timestamp para evitar sobrescribir datos anteriores
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.file_path = os.path.join(self.output_dir, f'dataset_{timestamp}.csv')
        
        self.csv_file = open(self.file_path, 'w', newline='')
        self.writer = csv.writer(self.csv_file)

        # Escribir cabecera
        header = [f'r{i}' for i in range(360)] + ['lin_vel', 'ang_vel', 'cmd_lin', 'cmd_ang']
        self.writer.writerow(header)

        # Variables para almacenar el estado
        self.scan = None
        self.odom_lin = 0.0
        self.odom_ang = 0.0
        self.cmd_lin = 0.0
        self.cmd_ang = 0.0
        self.last_save_time = rospy.Time.now()
        self.save_interval = rospy.Duration(0.1)  # Guardar datos a 10Hz

        # Suscripciones
        rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        rospy.Subscriber('/odom', Odometry, self.odom_cb)
        rospy.Subscriber('/cmd_vel', Twist, self.cmd_cb)

        # Publicar información de estado
        self.status_pub = rospy.Publisher('/data_collector/status', Twist, queue_size=1)
        self.rate = rospy.Rate(20)  # 20Hz para el bucle principal
        
        rospy.loginfo(f"Iniciando recolección de datos. Guardando en: {self.file_path}")
        self.run()

    def scan_cb(self, msg):
        """Callback para el escáner láser"""
        # Convertir a numpy array para manipulación eficiente
        ranges = np.array(msg.ranges)
        
        # Procesar datos inválidos (inf, nan, 0)
        ranges[np.isinf(ranges)] = 10.0  # Reemplazar infinitos por valor máximo
        ranges[np.isnan(ranges)] = 10.0  # Reemplazar NaN por valor máximo
        ranges[ranges == 0] = 10.0       # Reemplazar ceros por valor máximo
        
        # Limitar rango máximo
        self.scan = np.clip(ranges, 0.0, 10.0)

    def odom_cb(self, msg):
        """Callback para la odometría"""
        self.odom_lin = msg.twist.twist.linear.x
        self.odom_ang = msg.twist.twist.angular.z

    def cmd_cb(self, msg):
        """Callback para comandos de velocidad"""
        self.cmd_lin = msg.linear.x
        self.cmd_ang = msg.angular.z

    def run(self):
        """Bucle principal para recolección de datos"""
        saved_count = 0
        try:
            while not rospy.is_shutdown():
                current_time = rospy.Time.now()
                
                # Solo guardar datos si tenemos un escaneo y ha pasado suficiente tiempo
                if self.scan is not None and (current_time - self.last_save_time) > self.save_interval:
                    # Añadir un poco de ruido para mejorar la generalización
                    scan_processed = self.scan + np.random.normal(0, 0.01, self.scan.shape)
                    scan_processed = np.clip(scan_processed, 0.0, 10.0)
                    
                    # Guardar datos
                    row = list(scan_processed) + [self.odom_lin, self.odom_ang, self.cmd_lin, self.cmd_ang]
                    self.writer.writerow(row)
                    saved_count += 1
                    
                    # Actualizar tiempo de guardado
                    self.last_save_time = current_time
                    
                    # Mostrar estado cada 100 muestras
                    if saved_count % 100 == 0:
                        rospy.loginfo(f"Muestras guardadas: {saved_count}")
                        
                        # Publicar estado
                        status_msg = Twist()
                        status_msg.linear.x = float(saved_count)
                        status_msg.angular.z = float(saved_count) / 100.0  # Minutos aprox
                        self.status_pub.publish(status_msg)
                
                self.rate.sleep()
                
        except Exception as e:
            rospy.logerr(f"Error en recolección de datos: {e}")
        finally:
            self.csv_file.close()
            rospy.loginfo(f"Recolección finalizada. {saved_count} muestras guardadas en {self.file_path}")

if __name__ == '__main__':
    try:
        DataCollector()
    except rospy.ROSInterruptException:
        pass