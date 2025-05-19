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


class YOLODetector:
    def __init__(self, model_name='yolov5s', confidence_threshold=0.5):
        """Inicializar detector de objetos YOLO"""
        self.confidence_threshold = confidence_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        rospy.loginfo(f"Inicializando detector YOLO en {self.device}")
        try:
            # Cargar modelo YOLOv5 usando torch.hub
            self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
            self.model.to(self.device)
            self.model.eval()
            
            # Configurar el modelo
            self.model.conf = confidence_threshold  # Umbral de confianza
            self.model.classes = None  # Detectar todas las clases
            self.model.max_det = 10  # Máximo número de detecciones
            
            rospy.loginfo(f"Modelo YOLO {model_name} cargado correctamente")
        except Exception as e:
            rospy.logerr(f"Error al cargar el modelo YOLO: {e}")
            self.model = None
            
    def detect(self, image):
        """Detectar objetos en una imagen usando YOLO"""
        if self.model is None:
            return None
            
        try:
            # Realizar inferencia
            results = self.model(image)
            
            # Extraer predicciones
            predictions = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]
            
            # Filtrar por umbral de confianza
            detections = []
            for pred in predictions:
                x1, y1, x2, y2, conf, class_id = pred
                if conf >= self.confidence_threshold:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class_id': int(class_id),
                        'class_name': self.model.names[int(class_id)]
                    })
            
            return detections
        except Exception as e:
            rospy.logerr(f"Error en la detección YOLO: {e}")
            return None


class CNNYOLOController:
    def __init__(self):
        rospy.init_node('cnn_yolo_controller')
        
        # Cargar parámetros
        self.cnn_model_path = rospy.get_param('~cnn_model_path', 'model.pt')
        self.yolo_model = rospy.get_param('~yolo_model', 'yolov5s')
        self.yolo_confidence = rospy.get_param('~yolo_confidence', 0.5)
        self.max_range = rospy.get_param('~max_range', 10.0)
        self.max_lin_vel = rospy.get_param('~max_lin_vel', 0.5)  # m/s
        self.max_ang_vel = rospy.get_param('~max_ang_vel', 1.5)  # rad/s
        self.control_rate = rospy.get_param('~control_rate', 10)  # Hz
        self.camera_topic = rospy.get_param('~camera_topic', '/camera/rgb/image_raw')
        self.use_compressed = rospy.get_param('~use_compressed', False)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # OpenCV bridge
        self.bridge = CvBridge()
        
        # Inicializar modelos
        rospy.loginfo(f"Cargando modelo CNN desde: {self.cnn_model_path}")
        self.cnn_model = self.load_cnn_model()
        
        rospy.loginfo(f"Inicializando detector YOLO: {self.yolo_model}")
        self.yolo_detector = YOLODetector(
            model_name=self.yolo_model,
            confidence_threshold=self.yolo_confidence
        )
        
        # Publicadores y suscriptores
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.imu_pub = rospy.Publisher('/imu', Imu, queue_size=1)
        self.detection_pub = rospy.Publisher('/object_detection', String, queue_size=10)
        self.detection_image_pub = rospy.Publisher('/detection_image', Image, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan', LaserScan, self.scan_callback)
        
        # Suscriptor de la cámara
        if self.use_compressed:
            self.image_sub = rospy.Subscriber(
                f"{self.camera_topic}/compressed", 
                CompressedImage, 
                self.image_callback
            )
        else:
            self.image_sub = rospy.Subscriber(
                self.camera_topic, 
                Image, 
                self.image_callback
            )
        
        # Variables de estado
        self.latest_scan = None
        self.latest_image = None
        self.running = True
        
        rospy.loginfo("CNN-YOLO Controller inicializado")
        
    def load_cnn_model(self):
        """Cargar modelo CNN desde archivo"""
        try:
            # Comprobar si existe el archivo
            if not os.path.exists(self.cnn_model_path):
                rospy.logerr(f"El archivo de modelo CNN no existe: {self.cnn_model_path}")
                return None
                
            # Cargar modelo
            checkpoint = torch.load(self.cnn_model_path, map_location=self.device)
            
            # Inicializar modelo
            model = TurtlebotCNN().to(self.device)
            
            # Cargar estado del modelo
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()  # Establecer en modo evaluación
            rospy.loginfo(f"Modelo CNN cargado correctamente en {self.device}")
            return model
        except Exception as e:
            rospy.logerr(f"Error al cargar el modelo CNN: {e}")
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
    
    def image_callback(self, msg):
        """Callback para los datos de la cámara"""
        try:
            # Convertir mensaje ROS a imagen OpenCV
            if isinstance(msg, CompressedImage):
                np_arr = np.frombuffer(msg.data, np.uint8)
                cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            else:
                cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
                
            self.latest_image = cv_image
            
            # Procesar imagen con YOLO si está disponible
            self.process_image(cv_image)
        except CvBridgeError as e:
            rospy.logerr(f"Error en el puente CV: {e}")
        except Exception as e:
            rospy.logerr(f"Error en el procesamiento de la imagen: {e}")
    
    def process_image(self, image):
        """Procesar imagen con YOLO y publicar resultados"""
        if self.yolo_detector is None:
            return
            
        # Detectar objetos
        detections = self.yolo_detector.detect(image)
        
        if detections is None or len(detections) == 0:
            return
            
        # Dibujar resultados en la imagen
        vis_image = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            label = f"{det['class_name']} {det['confidence']:.2f}"
            
            # Dibujar rectángulo y etiqueta
            cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis_image, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Publicar imagen con detecciones
        try:
            detection_msg = self.bridge.cv2_to_imgmsg(vis_image, "bgr8")
            self.detection_image_pub.publish(detection_msg)
        except CvBridgeError as e:
            rospy.logerr(f"Error al convertir imagen: {e}")
        
        # Publicar información de detecciones
        detection_text = ""
        for i, det in enumerate(detections):
            detection_text += f"Objeto {i+1}: {det['class_name']} (conf: {det['confidence']:.2f})\n"
        
        self.detection_pub.publish(detection_text)
        
        # Imprimir detecciones ocasionalmente
        if rospy.get_time() % 5 < 0.1 and detections:
            rospy.loginfo(f"Detecciones: {len(detections)} objetos")
            for i, det in enumerate(detections[:3]):  # Mostrar solo las primeras 3 detecciones
                rospy.loginfo(f"  {i+1}. {det['class_name']} (conf: {det['confidence']:.2f})")
    
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
        if self.cnn_model is None:
            rospy.logerr("El modelo CNN no está cargado, no se pueden hacer predicciones")
            return 0.0, 0.0
            
        with torch.no_grad():
            scan_tensor = self.preprocess_scan(scan)
            output = self.cnn_model(scan_tensor)
            lin_vel, ang_vel = output[0].cpu().numpy()
            
        # Limitar velocidades
        lin_vel = np.clip(lin_vel, -self.max_lin_vel, self.max_lin_vel)
        ang_vel = np.clip(ang_vel, -self.max_ang_vel, self.max_ang_vel)
        
        return lin_vel, ang_vel
    
    def run(self):
        """Bucle principal de control"""
        rate = rospy.Rate(self.control_rate)
        
        rospy.loginfo("Iniciando control del robot y detección de objetos...")
        
        while not rospy.is_shutdown() and self.running:
            # Control basado en CNN
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
                
                # Log ocasional (cada 5 segundos)
                if rospy.get_time() % 5 < 0.1:
                    rospy.loginfo(f"Velocidad: lin={lin_vel:.2f} m/s, ang={ang_vel:.2f} rad/s")
            
            rate.sleep()

if __name__ == '__main__':
    try:
        controller = CNNYOLOController()
        controller.run()
    except rospy.ROSInterruptException:
        pass