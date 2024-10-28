import torch
import cv2
import numpy as np
import os
import yaml
import json
import logging
import datetime
import sqlite3
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
import threading
import queue
import time
from typing import List, Dict, Union, Tuple
import streamlit as st
from fastapi import FastAPI, File, UploadFile
import uvicorn

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('epp_detector.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Gestión de configuración del sistema"""
    def __init__(self, config_path: str = 'config.yaml'):
        self.config_path = config_path
        self.config = self.load_config()
        
    def load_config(self) -> dict:
        """Cargar configuración desde archivo YAML"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        return self.get_default_config()
    
    def get_default_config(self) -> dict:
        """Configuración por defecto"""
        return {
            'model': {
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'device': 'cuda' if torch.cuda.is_available() else 'cpu'
            },
            'classes': {
                0: 'casco',
                1: 'chaleco_reflectante',
                2: 'gafas_seguridad',
                3: 'guantes',
                4: 'botas_seguridad',
                5: 'mascarilla',
                6: 'proteccion_auditiva',
                7: 'arnes_seguridad'
            },
            'alerts': {
                'email': {
                    'enabled': True,
                    'smtp_server': 'smtp.gmail.com',
                    'smtp_port': 587,
                    'sender_email': '',
                    'sender_password': '',
                    'recipient_emails': []
                }
            },
            'database': {
                'path': 'epp_detections.db'
            }
        }

class DatabaseManager:
    """Gestión de base de datos para almacenar detecciones y estadísticas"""
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Inicializar tablas de la base de datos"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Tabla de detecciones
        c.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                image_path TEXT,
                location TEXT,
                employee_id TEXT,
                class_id INTEGER,
                confidence REAL,
                bbox TEXT,
                compliant BOOLEAN
            )
        ''')
        
        # Tabla de estadísticas
        c.execute('''
            CREATE TABLE IF NOT EXISTS statistics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE,
                location TEXT,
                total_detections INTEGER,
                compliant_detections INTEGER,
                non_compliant_detections INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_detection(self, detection_data: Dict):
        """Guardar una detección en la base de datos"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('''
            INSERT INTO detections 
            (image_path, location, employee_id, class_id, confidence, bbox, compliant)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            detection_data['image_path'],
            detection_data['location'],
            detection_data['employee_id'],
            detection_data['class_id'],
            detection_data['confidence'],
            json.dumps(detection_data['bbox']),
            detection_data['compliant']
        ))
        
        conn.commit()
        conn.close()

class AlertManager:
    """Gestión de alertas y notificaciones"""
    def __init__(self, config: dict):
        self.config = config
        self.alert_queue = queue.Queue()
        self.start_alert_worker()
    
    def start_alert_worker(self):
        """Iniciar worker thread para procesar alertas"""
        def worker():
            while True:
                alert = self.alert_queue.get()
                if alert is None:
                    break
                self.process_alert(alert)
                self.alert_queue.task_done()
                
        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()
    
    def send_alert(self, alert_data: Dict):
        """Enviar alerta al queue para procesamiento"""
        self.alert_queue.put(alert_data)
    
    def process_alert(self, alert_data: Dict):
        """Procesar alerta y enviar notificación"""
        if self.config['alerts']['email']['enabled']:
            self.send_email_alert(alert_data)
    
    def send_email_alert(self, alert_data: Dict):
        """Enviar alerta por email"""
        try:
            msg = MIMEMultipart()
            msg['Subject'] = f"Alerta de EPP - {alert_data['type']}"
            msg['From'] = self.config['alerts']['email']['sender_email']
            
            body = f"""
            Se ha detectado una violación de EPP:
            Ubicación: {alert_data['location']}
            Timestamp: {alert_data['timestamp']}
            EPP Faltante: {alert_data['missing_epps']}
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            # Adjuntar imagen si existe
            if 'image_path' in alert_data:
                with open(alert_data['image_path'], 'rb') as f:
                    img = MIMEImage(f.read())
                    msg.attach(img)
            
            # Enviar email
            with smtplib.SMTP(self.config['alerts']['email']['smtp_server'], 
                            self.config['alerts']['email']['smtp_port']) as server:
                server.starttls()
                server.login(
                    self.config['alerts']['email']['sender_email'],
                    self.config['alerts']['email']['sender_password']
                )
                
                for recipient in self.config['alerts']['email']['recipient_emails']:
                    msg['To'] = recipient
                    server.send_message(msg)
                    
        except Exception as e:
            logger.error(f"Error sending email alert: {str(e)}")

class EPPDataset(Dataset):
    """Dataset personalizado para entrenamiento"""
    def __init__(self, 
                 img_dir: str, 
                 annot_dir: str, 
                 transform: A.Compose = None,
                 phase: str = 'train'):
        self.img_dir = img_dir
        self.annot_dir = annot_dir
        self.transform = transform or self.get_default_transform(phase)
        self.images = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    def get_default_transform(self, phase: str) -> A.Compose:
        """Obtener transformaciones por defecto"""
        if phase == 'train':
            return A.Compose([
                A.RandomBrightnessContrast(p=0.5),
                A.HueSaturationValue(p=0.5),
                A.ShiftScaleRotate(p=0.5),
                A.Blur(p=0.3),
                A.CLAHE(p=0.3),
                A.RandomResizedCrop(height=640, width=640, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))
        else:
            return A.Compose([
                A.Resize(height=640, width=640),
                A.Normalize(),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['labels']))
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        annot_path = os.path.join(self.annot_dir, img_name.replace('.jpg', '.txt'))
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        
        if os.path.exists(annot_path):
            with open(annot_path, 'r') as f:
                for line in f.readlines():
                    class_id, x, y, w, h = map(float, line.strip().split())
                    boxes.append([x, y, w, h])
                    labels.append(class_id)
        
        boxes = np.array(boxes)
        labels = np.array(labels)
        
        if self.transform:
            augmented = self.transform(image=image, bboxes=boxes, labels=labels)
            image = augmented['image']
            boxes = torch.tensor(augmented['bboxes'])
            labels = torch.tensor(augmented['labels'])
        
        return image, boxes, labels

class EPPDetector:
    """Clase principal para detección de EPPs"""
    def __init__(self, 
                 weights_path: str,
                 config: dict,
                 db_manager: DatabaseManager,
                 alert_manager: AlertManager):
        self.model = self.load_model(weights_path)
        self.config = config
        self.db_manager = db_manager
        self.alert_manager = alert_manager
        
    def load_model(self, weights_path: str) -> torch.nn.Module:
        """Cargar modelo pre-entrenado"""
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path)
        model.to(self.config['model']['device'])
        model.eval()
        return model
    
    def detect(self, 
              image_path: str, 
              location: str = None,
              employee_id: str = None) -> List[Dict]:
        """
        Realizar detección de EPPs en imagen
        """
        try:
            results = self.model(image_path)
            detections = []
            
            # Procesar detecciones
            for *xyxy, conf, cls in results.xyxy[0]:
                if conf >= self.config['model']['confidence_threshold']:
                    x1, y1, x2, y2 = map(int, xyxy)
                    detection = {
                        'image_path': image_path,
                        'location': location,
                        'employee_id': employee_id,
                        'class_id': int(cls),
                        'confidence': float(conf),
                        'bbox': [x1, y1, x2, y2],
                        'compliant': True
                    }
                    
                    detections.append(detection)
                    self.db_manager.save_detection(detection)
            
            # Verificar cumplimiento de EPPs
            self.check_compliance(detections, image_path, location)
            
            return detections
            
        except Exception as e:
            logger.error(f"Error in detection: {str(e)}")
            return []
    
    def check_compliance(self, 
                        detections: List[Dict],
                        image_path: str,
                        location: str):
        """Verificar cumplimiento de EPPs requeridos"""
        detected_classes = set(d['class_id'] for d in detections)
        required_classes = {0, 1}  # Casco y chaleco son obligatorios
        
        missing_epps = required_classes - detected_classes
        if missing_epps:
            alert_data = {
                'type': 'NON_COMPLIANT',
                'location': location,
                'timestamp': datetime.datetime.now().isoformat(),
                'missing_epps': [self.config['classes'][cls_id] for cls_id in missing_epps],
                'image_path': image_path
            }
            self.alert_manager.send_alert(alert_data)
    
    def process_video(self, 
                     video_path: str,
                     output_path: str = None,
                     location: str = None) -> str:
        """Procesar video para detección de EPPs"""
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise ValueError("Error opening video file")
            
            # Configurar writer si se especifica output_path
            if output_path:
                frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Guardar frame temporalmente
                temp_frame_path = 'temp_frame.jpg'
                cv2.imwrite(temp_frame_path, frame)
                
                # Realizar detección
                detections = self.detect(temp_frame_path, location)
                
                # Dibujar detecciones
                frame_with_dets = self.draw_detections(frame, detections)
                
                if output_path:
                    out.write(frame_with_dets)
                
                # Eliminar frame temporal
                os.remove(temp_frame_path)
            
            cap.release()
            if output_path:
                out.release()
                return output_path
                
        except Exception as e:
            logger.error(f"Error processing video: {str(e)}")
            return None
    
    def draw_detections(self, 
                       image: np.ndarray,
                       detections: List[Dict]) -> np.ndarray:
        """
