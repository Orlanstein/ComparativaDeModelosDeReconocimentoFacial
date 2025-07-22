#!/usr/bin/env python3
import cv2
import time
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from face_hog import EnhancedFaceTracking
from face_haar import HaarFaceTracker
from face_mtcnn import MTCNNFaceTracker

class SequentialFaceDetectionBenchmark:
    def __init__(self, sample_time=10, output_dir="resultados"):
        self.sample_time = sample_time  # Tiempo de muestreo en segundos
        self.output_dir = output_dir
        
        # Definir los detectores a evaluar en orden
        self.detectors = [
            ("HOG", EnhancedFaceTracking),
            ("HAAR", HaarFaceTracker),
            ("MTCNN", MTCNNFaceTracker)
        ]
        
        # Datos de métricas
        self.metrics_data = {}
        self.start_time = None
        self.current_run = 0
        
        # Crear directorio de resultados si no existe
        os.makedirs(self.output_dir, exist_ok=True)
        
    def find_next_run_number(self):
        """Encuentra el próximo número de ejecución disponible"""
        run_number = 1
        while True:
            dir_name = f"comparacion{run_number}"
            full_path = os.path.join(self.output_dir, dir_name)
            if not os.path.exists(full_path):
                return run_number
            run_number += 1
    
    def setup_output_directory(self):
        """Crea la estructura de directorios para los resultados"""
        self.current_run = self.find_next_run_number()
        self.run_dir = os.path.join(self.output_dir, f"comparacion{self.current_run}")
        os.makedirs(self.run_dir)
        
        # Subdirectorios
        os.makedirs(os.path.join(self.run_dir, "frames"))
        os.makedirs(os.path.join(self.run_dir, "metrics"))
    
    def run_detector_test(self, detector_name, detector_class):
        """Ejecuta una prueba para un solo detector"""
        print(f"\nIniciando prueba para {detector_name}...")
        
        # Inicializar detector
        if detector_name == "MTCNN":
            detector = detector_class(min_confidence=0.85)
        else:
            detector = detector_class()
        
        # Preparar estructura para métricas
        self.metrics_data[detector_name] = []
        start_time = time.time()
        
        try:
            while (time.time() - start_time) < self.sample_time:
                # Procesar frame
                frame, metrics = detector.process_frame()
                
                if frame is None:
                    continue
                
                # Registrar métricas
                metrics['timestamp'] = time.time() - start_time
                self.metrics_data[detector_name].append(metrics)
                
                # Mostrar vista previa
                cv2.imshow(f'{detector_name} Face Detection', frame)
                
                # Guardar frame de muestra periódicamente
                if len(self.metrics_data[detector_name]) % 5 == 0:
                    frame_path = os.path.join(self.run_dir, "frames", 
                                            f"{detector_name}_{len(self.metrics_data[detector_name])}.jpg")
                    cv2.imwrite(frame_path, frame)
                
                # Manejar interrupción de usuario
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print(f"Prueba de {detector_name} interrumpida por el usuario")
                    break
                
        except Exception as e:
            print(f"Error durante la prueba de {detector_name}: {str(e)}")
        finally:
            # Liberar recursos del detector
            if hasattr(detector, 'cap'):
                detector.cap.release()
            cv2.destroyAllWindows()
            
            # Guardar métricas para este detector
            if self.metrics_data[detector_name]:
                csv_path = os.path.join(self.run_dir, "metrics", f"{detector_name}_metrics.csv")
                df = pd.DataFrame(self.metrics_data[detector_name])
                df.to_csv(csv_path, index=False)
                print(f"Resultados de {detector_name} guardados en {csv_path}")
    
    def run_benchmark(self):
        """Ejecuta el benchmark de comparación secuencial"""
        self.setup_output_directory()
        print(f"Benchmark iniciado. Cada prueba durará {self.sample_time} segundos.")
        
        # Ejecutar pruebas secuenciales
        for detector_name, detector_class in self.detectors:
            self.run_detector_test(detector_name, detector_class)
        
        # Generar reporte comparativo
        self.create_summary_report()
        self.generate_comparison_plots()
        
        print(f"\nBenchmark completado. Resultados guardados en {self.run_dir}")
    
    def create_summary_report(self):
        """Crea un reporte resumen comparativo"""
        summary_data = []
        
        for detector_name, metrics_list in self.metrics_data.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                summary_data.append({
                    'Modelo': detector_name,
                    'FPS Promedio': df['fps'].mean(),
                    'FPS Maximo': df['fps'].max(),
                    'FPS Minimo': df['fps'].min(),
                    'Detecciones Promedio': df['face_count'].mean(),
                    'CPU Promedio': df['cpu_usage'].mean(),
                    'RAM Promedio': df['mem_usage'].mean(),
                    'Muestras': len(metrics_list)
                })
        
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(self.run_dir, "comparative_summary.csv")
            summary_df.to_csv(summary_path, index=False)
            
            # También guardar como markdown para mejor legibilidad
            md_path = os.path.join(self.run_dir, "RESULTADOS.md")
            with open(md_path, 'w') as f:
                f.write("# Resultados de Comparacion de Modelos de Detección Facial\n\n")
                f.write(f"Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Duracion por prueba: {self.sample_time} segundos\n\n")
                f.write("## Resumen Comparativo\n\n")
                f.write(summary_df.to_markdown(index=False))
                f.write("\n\n## Metricas Detalladas\n")
                f.write("Los datos completos están disponibles en los archivos CSV dentro de la carpeta metrics/")
    
    def generate_comparison_plots(self):
        """Genera gráficos comparativos de las métricas"""
        plt.style.use('seaborn')
        
        # Gráfico de FPS
        plt.figure(figsize=(12, 7))
        for name, metrics_list in self.metrics_data.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                plt.plot(df['timestamp'], df['fps'], label=name, linewidth=2)
        plt.title('Comparación de Rendimiento (FPS)', fontsize=14)
        plt.xlabel('Tiempo (s)', fontsize=12)
        plt.ylabel('FPS', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'fps_comparison.png'), dpi=300)
        plt.close()
        
        # Gráfico de detecciones
        plt.figure(figsize=(12, 7))
        for name, metrics_list in self.metrics_data.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                plt.plot(df['timestamp'], df['face_count'], label=name, linewidth=2)
        plt.title('Comparación de Detecciones de Rostros', fontsize=14)
        plt.xlabel('Tiempo (s)', fontsize=12)
        plt.ylabel('Número de Rostros', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'face_detection_comparison.png'), dpi=300)
        plt.close()
        
        # Gráfico de uso de CPU
        plt.figure(figsize=(12, 7))
        for name, metrics_list in self.metrics_data.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                plt.plot(df['timestamp'], df['cpu_usage'], label=name, linewidth=2)
        plt.title('Comparación de Uso de CPU', fontsize=14)
        plt.xlabel('Tiempo (s)', fontsize=12)
        plt.ylabel('Uso de CPU (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'cpu_usage_comparison.png'), dpi=300)
        plt.close()
        
        # Gráfico de uso de memoria
        plt.figure(figsize=(12, 7))
        for name, metrics_list in self.metrics_data.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                plt.plot(df['timestamp'], df['mem_usage'], label=name, linewidth=2)
        plt.title('Comparación de Uso de Memoria', fontsize=14)
        plt.xlabel('Tiempo (s)', fontsize=12)
        plt.ylabel('Uso de Memoria (%)', fontsize=12)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'memory_usage_comparison.png'), dpi=300)
        plt.close()

if __name__ == '__main__':
    # Configuración del benchmark
    benchmark = SequentialFaceDetectionBenchmark(
        sample_time=30,  # 30 segundos por prueba
        output_dir="resultados"
    )
    
    # Ejecutar comparación secuencial
    benchmark.run_benchmark()