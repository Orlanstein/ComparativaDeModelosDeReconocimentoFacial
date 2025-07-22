#!/usr/bin/env python3
import time
import psutil
import numpy as np

class PerformanceMonitor:
    def __init__(self):
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        self.cpu_usage = 0
        self.memory = 0
        self.face_counts = []
        
    def update(self, face_count=0):
        self.frame_count += 1
        self.face_counts.append(face_count)
        
        # Actualizar FPS cada segundo
        current_time = time.time()
        if current_time - self.last_time >= 1.0:
            self.fps = self.frame_count
            self.frame_count = 0
            self.last_time = current_time
            
            # Métricas del sistema
            self.cpu_usage = psutil.cpu_percent()
            self.memory = psutil.virtual_memory().percent
            
    def get_metrics(self):
        return {
            'fps': self.fps,
            'cpu': self.cpu_usage,
            'memory': self.memory,
            'avg_faces': np.mean(self.face_counts) if self.face_counts else 0
        }
