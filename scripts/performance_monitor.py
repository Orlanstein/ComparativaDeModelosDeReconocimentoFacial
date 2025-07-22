#!/usr/bin/env python3
import rospy
from std_msgs.msg import String
import pandas as pd
import time
import os
from datetime import datetime
import atexit
import sys

class PerformanceMonitor:
    def __init__(self):
        rospy.init_node('performance_monitor', anonymous=True)
        
        # Configuración de muestras
        self.samples_per_method = 100
        self.collected_samples = {
            'HOG': 0,
            'CNN': 0,
            'HAAR': 0,
            'DLIB': 0,
            'MFN': 0
        }
        self.all_methods_completed = False
        
        # Suscriptores
        self.subscribers = [
            rospy.Subscriber('/hog_face_status', String, self.callback, callback_args='HOG'),
            rospy.Subscriber('/cnn_face_status', String, self.callback, callback_args='CNN'),
            rospy.Subscriber('/haar_face_status', String, self.callback, callback_args='HAAR'),
            rospy.Subscriber('/dlib_face_status', String, self.callback, callback_args='DLIB'),
            rospy.Subscriber('/mfn_face_status', String, self.callback, callback_args='MFN')
        ]
        
        # Datos
        self.data = []
        self.results_dir = os.path.expanduser('~/catkin_ws/src/pony/results')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Registrar función de guardado
        atexit.register(self.save_results)
        rospy.on_shutdown(self.save_results)
        
        rospy.loginfo(f"Monitor listo. Colectando {self.samples_per_method} muestras por método...")

    def callback(self, msg, model):
        if self.all_methods_completed:
            return
            
        try:
            # Verificar si ya tenemos suficientes muestras para este método
            if self.collected_samples[model] >= self.samples_per_method:
                return
                
            parts = msg.data.split('|')
            if len(parts) >= 5:
                entry = {
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    'model': model,
                    'fps': float(parts[1].split(':')[1]),
                    'faces': int(parts[2].split(':')[1]),
                    'cpu': float(parts[3].split(':')[1]),
                    'mem': float(parts[4].split(':')[1])
                }
                self.data.append(entry)
                self.collected_samples[model] += 1
                
                rospy.loginfo(f"Dato recibido ({model}): Muestra {self.collected_samples[model]}/{self.samples_per_method} - "
                              f"FPS: {entry['fps']:.1f}, Caras: {entry['faces']}, CPU: {entry['cpu']:.1f}%, Mem: {entry['mem']:.1f}%")
                
                # Verificar si hemos completado todas las muestras
                self.check_completion()
                
        except Exception as e:
            rospy.logerr(f"Error procesando datos: {str(e)}")

    def check_completion(self):
        # Verificar si todos los métodos han alcanzado el número de muestras requerido
        all_completed = all(count >= self.samples_per_method for count in self.collected_samples.values())
        
        if all_completed and not self.all_methods_completed:
            self.all_methods_completed = True
            rospy.loginfo("\n¡Todas las muestras recolectadas! Guardando resultados y cerrando...")
            self.save_results()
            rospy.signal_shutdown("Recolección completada")
            sys.exit(0)

    def save_results(self):
        if not self.data:
            rospy.logwarn("No hay datos para guardar")
            return
            
        try:
            # Crear DataFrames
            df_raw = pd.DataFrame(self.data)
            
            # Calcular estadísticas
            stats = df_raw.groupby('model').agg({
                'fps': ['mean', 'max', 'min', 'std'],
                'faces': ['mean', 'max'],
                'cpu': 'mean',
                'mem': 'mean'
            }).round(2)
            
            # Renombrar columnas
            stats.columns = ['FPS Promedio', 'FPS Máximo', 'FPS Mínimo', 'FPS Desviación',
                           'Detección Promedio', 'Detección Máxima',
                           'CPU Promedio', 'Memoria Promedio']
            
            # Generar timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Guardar archivos
            raw_file = os.path.join(self.results_dir, f'raw_data_{timestamp}.csv')
            stats_file = os.path.join(self.results_dir, f'stats_{timestamp}.csv')
            report_file = os.path.join(self.results_dir, f'summary_{timestamp}.txt')
            
            df_raw.to_csv(raw_file, index=False)
            stats.to_csv(stats_file)
            
            # Crear reporte manualmente
            with open(report_file, 'w') as f:
                f.write("=== RESUMEN COMPARATIVO ===\n\n")
                f.write("Método\t\tMuestras\tFPS (Avg±Std)\tFPS Range\t\tCaras (Avg)\tCPU (%)\tMem (%)\n")
                f.write("="*100 + "\n")
                
                for model in stats.index:
                    row = stats.loc[model]
                    f.write(f"{model}\t\t{self.samples_per_method}\t\t"
                            f"{row['FPS Promedio']}±{row['FPS Desviación']}\t"
                            f"{row['FPS Mínimo']}-{row['FPS Máximo']}\t\t"
                            f"{row['Detección Promedio']}\t\t"
                            f"{row['CPU Promedio']}\t"
                            f"{row['Memoria Promedio']}\n")
                
                f.write("\n=== DATOS COMPLETOS ===\n")
                f.write(f"Datos crudos: {raw_file}\n")
                f.write(f"Estadísticas: {stats_file}\n")
            
            rospy.loginfo(f"\nResumen guardado en: {report_file}")
            rospy.loginfo(f"Datos completos en: {raw_file}")
            rospy.loginfo(f"Estadísticas en: {stats_file}")
            
            # Mostrar resumen en consola
            rospy.loginfo("\n=== RESUMEN DE RESULTADOS ===")
            rospy.loginfo("Método\t\tMuestras\tFPS (Avg±Std)\tFPS Range\t\tCaras (Avg)\tCPU (%)\tMem (%)")
            rospy.loginfo("="*100)
            for model in stats.index:
                row = stats.loc[model]
                rospy.loginfo(f"{model}\t\t{self.samples_per_method}\t\t"
                              f"{row['FPS Promedio']}±{row['FPS Desviación']}\t"
                              f"{row['FPS Mínimo']}-{row['FPS Máximo']}\t\t"
                              f"{row['Detección Promedio']}\t\t"
                              f"{row['CPU Promedio']}\t"
                              f"{row['Memoria Promedio']}")
            
        except Exception as e:
            rospy.logerr(f"Error al guardar: {str(e)}")

if __name__ == '__main__':
    try:
        monitor = PerformanceMonitor()
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Interrupción por teclado recibida. Guardando datos recolectados...")
        monitor.save_results()
    except Exception as e:
        rospy.logerr(f"Error inesperado: {str(e)}")
