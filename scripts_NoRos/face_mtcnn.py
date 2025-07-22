#!/usr/bin/env python3
import cv2
import time
import psutil
from mtcnn import MTCNN

class MTCNNFaceTracker:
    def __init__(self, camera_index=None, min_confidence=0.9):
        # Configuración de cámara
        self.camera_index = camera_index if camera_index is not None else self._detect_camera()
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Inicializar detector MTCNN
        self.detector = MTCNN()
        self.min_confidence = min_confidence
        self.min_face_size = 80
        
        # Variables de rendimiento
        self.frame_count = 0
        self.fps = 0
        self.cpu_usage = 0
        self.mem_usage = 0
        self.face_count = 0
        self.last_time = time.time()
        
        print("MTCNN Face Tracker iniciado correctamente")

    def _detect_camera(self):
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return i
        return 0

    def _detect_faces(self, frame):
        try:
            # Convertir a RGB (MTCNN espera imágenes RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = self.detector.detect_faces(rgb_frame)
            
            # Filtrar por confianza y tamaño
            face_locations = []
            for det in detections:
                if det['confidence'] >= self.min_confidence:
                    x, y, w, h = det['box']
                    if w >= self.min_face_size and h >= self.min_face_size:
                        face_locations.append((y, x+w, y+h, x))
            
            return face_locations
        except Exception as e:
            print(f"Error en detección: {str(e)}")
            return []

    def draw_metrics(self, frame):
        # Configuración de texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        color = (0, 255, 0)  # Verde (para MTCNN)
        y_offset = 30
        line_height = 30
        
        # Texto a mostrar
        metrics = [
            f"FPS: {self.fps:.1f}",
            f"CPU: {self.cpu_usage:.1f}%",
            f"MEM: {self.mem_usage:.1f}%",
            f"Faces: {self.face_count}"
        ]
        
        # Dibujar cada métrica
        for i, text in enumerate(metrics):
            cv2.putText(frame, text, (10, y_offset + i * line_height), 
                       font, font_scale, color, font_thickness, cv2.LINE_AA)

    def get_metrics(self):
        """Devuelve un diccionario con las métricas actuales"""
        return {
            'fps': self.fps,
            'cpu_usage': self.cpu_usage,
            'mem_usage': self.mem_usage,
            'face_count': self.face_count,
            'detector': 'MTCNN'
        }

    def process_frame(self):
        """Procesa un solo frame y devuelve el frame con las detecciones y las métricas"""
        ret, frame = self.cap.read()
        if not ret:
            print("Warning: Error al capturar frame")
            return None, None
        
        face_locations = self._detect_faces(frame)
        self.face_count = len(face_locations)
        
        # Calcular métricas
        self.frame_count += 1
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_time)
            self.frame_count = 0
            self.last_time = current_time
            
            # Obtener uso de recursos
            self.cpu_usage = psutil.cpu_percent()
            self.mem_usage = psutil.virtual_memory().percent
        
        # Dibujar detecciones
        for (top, right, bottom, left) in face_locations:
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        
        # Dibujar métricas en el frame
        self.draw_metrics(frame)
        
        return frame, self.get_metrics()

    def run(self):
        """Ejecuta el bucle principal de detección (para uso independiente)"""
        try:
            while True:
                frame, metrics = self.process_frame()
                if frame is None:
                    continue
                
                cv2.imshow('Face Tracking MTCNN', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error crítico: {str(e)}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = MTCNNFaceTracker()
    tracker.run()