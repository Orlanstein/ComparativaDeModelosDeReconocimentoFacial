#!/usr/bin/env python3
import cv2
import time
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from face_hog import EnhancedFaceTracking
from face_haar import HaarFaceTracker
from face_mtcnn import MTCNNFaceTracker

class SequentialFaceDetectionBenchmark:
    def __init__(self, sample_time=10, output_dir="resultados", camera_index=0):
        self.sample_time = sample_time  # Tiempo de muestreo en segundos
        self.output_dir = output_dir
        self.camera_index = camera_index  # Índice de la cámara a usar
        
        # Definir los detectores a evaluar en orden
        self.detectors = [
            ("HOG", EnhancedFaceTracking),
            ("HAAR", HaarFaceTracker),
            ("MTCNN", MTCNNFaceTracker)
        ]
        
        # Datos de métricas
        self.metrics_data = {}
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
        
        # Subdirectorio para métricas
        self.metrics_dir = os.path.join(self.run_dir, "metrics")
        os.makedirs(self.metrics_dir)
    
    def run_detector_test(self, detector_name, detector_class):
        """Ejecuta una prueba para un solo detector"""
        print(f"\nIniciando prueba para {detector_name} (Cámara {self.camera_index})...")
        
        # Inicializar detector con el índice de cámara especificado
        if detector_name == "MTCNN":
            detector = detector_class(min_confidence=0.85, camera_index=self.camera_index)
        else:
            detector = detector_class(camera_index=self.camera_index)
        
        # Preparar estructura para métricas
        self.metrics_data[detector_name] = []
        start_time = time.time()
        last_metrics_time = start_time
        
        try:
            while (time.time() - start_time) < self.sample_time:
                # Procesar frame
                frame, metrics = detector.process_frame()
                
                if frame is None:
                    continue
                
                # Actualizar métricas cada 0.1 segundos aproximadamente
                current_time = time.time()
                if current_time - last_metrics_time > 0.1:
                    metrics['timestamp'] = current_time - start_time
                    self.metrics_data[detector_name].append(metrics)
                    last_metrics_time = current_time
                
                # Mostrar vista previa
                cv2.imshow(f'{detector_name} Face Detection', frame)
                
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
                csv_path = os.path.join(self.metrics_dir, f"{detector_name}_metrics.csv")
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
                f.write("\n\n## Métricas Detalladas\n")
                f.write("Los datos completos estan disponibles en los archivos CSV dentro de la carpeta metrics/\n")
                f.write("Graficas comparativas disponibles en el mismo directorio.")
    
    def generate_comparison_plots(self):
        """Genera gráficos comparativos de las métricas"""
        plt.style.use('seaborn-v0_8')
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['figure.figsize'] = (14, 8)
        
        # Gráfico de FPS
        plt.figure()
        for name, metrics_list in self.metrics_data.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                # Suavizar datos con media móvil
                df['fps_smoothed'] = df['fps'].rolling(window=5, min_periods=1).mean()
                plt.plot(df['timestamp'], df['fps_smoothed'], label=name, linewidth=2.5)
        
        plt.title('Comparación de Rendimiento (FPS)', fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('FPS')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'fps_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Gráfico de detecciones
        plt.figure()
        for name, metrics_list in self.metrics_data.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                plt.plot(df['timestamp'], df['face_count'], label=name, linewidth=2.5)
        
        plt.title('Comparación de Detecciones de Rostros', fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Número de Rostros')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'face_detection_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Gráfico de uso de CPU
        plt.figure()
        for name, metrics_list in self.metrics_data.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                # Suavizar datos con media móvil
                df['cpu_smoothed'] = df['cpu_usage'].rolling(window=5, min_periods=1).mean()
                plt.plot(df['timestamp'], df['cpu_smoothed'], label=name, linewidth=2.5)
        
        plt.title('Comparación de Uso de CPU', fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Uso de CPU (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'cpu_usage_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Gráfico de uso de memoria
        plt.figure()
        for name, metrics_list in self.metrics_data.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                # Suavizar datos con media móvil
                df['mem_smoothed'] = df['mem_usage'].rolling(window=5, min_periods=1).mean()
                plt.plot(df['timestamp'], df['mem_smoothed'], label=name, linewidth=2.5)
        
        plt.title('Comparación de Uso de Memoria', fontweight='bold')
        plt.xlabel('Tiempo (s)')
        plt.ylabel('Uso de Memoria (%)')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'memory_usage_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Gráfico de dispersión FPS vs Detecciones
        plt.figure()
        for name, metrics_list in self.metrics_data.items():
            if metrics_list:
                df = pd.DataFrame(metrics_list)
                # Filtrar valores extremos
                df = df[(df['fps'] > 1) & (df['fps'] < 100)]
                plt.scatter(df['face_count'], df['fps'], label=name, alpha=0.6, s=80)
        
        plt.title('Relación entre Detecciones y Rendimiento (FPS)', fontweight='bold')
        plt.xlabel('Número de Rostros Detectados')
        plt.ylabel('FPS')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_dir, 'fps_vs_detections.png'), dpi=150, bbox_inches='tight')
        plt.close()
        
        # Gráfico de barras comparativas
        summary_path = os.path.join(self.run_dir, "comparative_summary.csv")
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
            
            # Gráfico de barras para FPS promedio
            plt.figure()
            plt.bar(summary_df['Modelo'], summary_df['FPS Promedio'], color=['blue', 'green', 'red'])
            plt.title('FPS Promedio por Modelo', fontweight='bold')
            plt.ylabel('FPS')
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'average_fps_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Gráfico de barras para uso de recursos
            fig, ax = plt.subplots()
            width = 0.35
            x = np.arange(len(summary_df))
            
            rects1 = ax.bar(x - width/2, summary_df['CPU Promedio'], width, label='CPU')
            rects2 = ax.bar(x + width/2, summary_df['RAM Promedio'], width, label='RAM')
            
            ax.set_title('Uso Promedio de Recursos', fontweight='bold')
            ax.set_ylabel('Porcentaje (%)')
            ax.set_xticks(x)
            ax.set_xticklabels(summary_df['Modelo'])
            ax.legend()
            ax.grid(axis='y', linestyle='--', alpha=0.7)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.run_dir, 'resource_usage_comparison.png'), dpi=150, bbox_inches='tight')
            plt.close()

if __name__ == '__main__':
    # Configuración del benchmark
    camera_index = 0  # Cambia este valor para usar otra cámara (0 es la predeterminada)
    
    benchmark = SequentialFaceDetectionBenchmark(
        sample_time=30,  # 30 segundos por prueba
        output_dir="resultados",
        camera_index=camera_index
    )
    
    # Ejecutar comparación secuencial
    benchmark.run_benchmark()