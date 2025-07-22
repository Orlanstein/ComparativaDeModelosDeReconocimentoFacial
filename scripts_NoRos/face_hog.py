#!/usr/bin/env python3
import cv2
import face_recognition
import time
import psutil

class EnhancedFaceTracking:
    def __init__(self, camera_index=None):
        # Configuración de cámara
        self.camera_index = camera_index if camera_index is not None else self.detect_camera()
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            print("Error: No se pudo abrir la cámara")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        
        # Parámetros de detección
        self.detection_scale = 0.5
        self.min_face_size = 80
        
        # Variables de rendimiento
        self.frame_count = 0
        self.fps = 0
        self.cpu_usage = 0
        self.mem_usage = 0
        self.face_count = 0
        self.last_time = time.time()
        
        print("Tracker HOG iniciado")

    def detect_camera(self):
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return i
        return 0

    def adaptive_face_detection(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=self.detection_scale, fy=self.detection_scale)
        rgb_small_frame = small_frame[:, :, ::-1]
        
        face_locations = face_recognition.face_locations(
            rgb_small_frame,
            model='hog',
            number_of_times_to_upsample=0
        )
        
        # Filtrar y escalar detecciones
        filtered_locations = []
        scale_factor = 1 / self.detection_scale
        for (top, right, bottom, left) in face_locations:
            face_height = (bottom - top) * scale_factor
            if face_height >= self.min_face_size:
                filtered_locations.append((
                    int(top * scale_factor),
                    int(right * scale_factor),
                    int(bottom * scale_factor),
                    int(left * scale_factor)
                ))
        
        return filtered_locations

    def draw_metrics(self, frame):
        # Configuración de texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        color = (0, 0, 255)  # Rojo (para coincidir con el color de los rectángulos)
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
            'face_count': self.face_count
        }

    def process_frame(self):
        """Procesa un solo frame y devuelve el frame con las detecciones y las métricas"""
        ret, frame = self.cap.read()
        if not ret:
            print("Warning: Error al capturar frame")
            return None, None
        
        face_locations = self.adaptive_face_detection(frame)
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
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        
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
                
                cv2.imshow('Face Tracking HOG', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = EnhancedFaceTracking()
    tracker.run()