#!/usr/bin/env python3
import cv2
import rospy
import dlib
from std_msgs.msg import String
import time
import psutil

class DlibFaceTracker:
    def __init__(self):
        rospy.init_node('dlib_face_tracker')
        self.pub = rospy.Publisher('/dlib_face_status', String, queue_size=10)
        
        # Configuración de cámara
        self.camera_index = self.detect_camera()
        self.cap = cv2.VideoCapture(self.camera_index)
        
        if not self.cap.isOpened():
            rospy.logerr("No se pudo abrir la cámara")
            rospy.signal_shutdown("Error de cámara")
            return
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        
        # Inicializar detector Dlib
        self.detector = dlib.get_frontal_face_detector()
        self.min_face_size = 80
        
        # Variables de rendimiento
        self.frame_count = 0
        self.fps = 0
        self.cpu_usage = 0
        self.mem_usage = 0
        self.face_count = 0
        self.last_time = time.time()
        
        rospy.loginfo("Dlib Face Tracker iniciado")

    def detect_camera(self):
        for i in range(3):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                cap.release()
                return i
        return 0

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray, 0)  # 0 = sin upsampling
        
        # Convertir a formato (top, right, bottom, left)
        face_locations = []
        for face in faces:
            if face.height() >= self.min_face_size:
                face_locations.append((
                    face.top(),
                    face.right(),
                    face.bottom(),
                    face.left()
                ))
        
        return face_locations

    def draw_metrics(self, frame):
        # Configuración de texto
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        color = (0, 255, 255)  # Amarillo (para coincidir con el color de los rectángulos)
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

    def run(self):
        try:
            while not rospy.is_shutdown():
                start_time = time.time()
                
                ret, frame = self.cap.read()
                if not ret:
                    rospy.logwarn("Error al capturar frame")
                    continue
                
                face_locations = self.detect_faces(frame)
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
                    
                    # Publicar métricas
                    msg = f"DLIB|FPS:{self.fps:.1f}|Faces:{self.face_count}|CPU:{self.cpu_usage:.1f}|MEM:{self.mem_usage:.1f}"
                    self.pub.publish(msg)
                    rospy.loginfo(f"Publicado: {msg}")
                
                # Mostrar resultados
                for (top, right, bottom, left) in face_locations:
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                
                # Dibujar métricas en el frame
                self.draw_metrics(frame)
                
                cv2.imshow('Face Tracking DLIB', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        except Exception as e:
            rospy.logerr(f"Error: {str(e)}")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()

if __name__ == '__main__':
    tracker = DlibFaceTracker()
    tracker.run()
