#!/usr/bin/env python3
import pandas as pd
import os
import argparse
from datetime import datetime

# Fallback para colored si termcolor no está disponible
try:
    from termcolor import colored
except ImportError:
    def colored(text, color=None, attrs=None):
        return text

class SimpleModelSelector:
    def __init__(self, results_dir):
        self.results_dir = os.path.expanduser(results_dir)
        self.weights = {
            'fps': 0.4,    # Mayor es mejor
            'cpu': -0.3,   # Menor es mejor
            'mem': -0.2,   # Menor es mejor
            'faces': 0.1   # Mayor es mejor
        }

    def load_data(self):
        """Carga el archivo de datos más reciente"""
        try:
            csv_files = [f for f in os.listdir(self.results_dir) 
                        if f.startswith('raw_data') and f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No se encontraron archivos CSV")
            
            latest_file = max(csv_files, key=lambda x: datetime.strptime(x[9:-4], "%Y%m%d_%H%M%S"))
            file_path = os.path.join(self.results_dir, latest_file)
            
            return pd.read_csv(file_path)
        except Exception as e:
            print(colored(f"\nError al cargar datos: {str(e)}", 'red'))
            return None

    def analyze_data(self, df):
        """Realiza el análisis completo de los datos"""
        if df is None or df.empty:
            return None

        # Calcular estadísticas básicas
        stats = df.groupby('model').agg({
            'fps': ['mean', 'std'],
            'cpu': 'mean',
            'mem': 'mean',
            'faces': 'mean'
        })

        # Simplificar la estructura del MultiIndex
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        
        # Normalización manual (0-1)
        for metric in ['fps', 'cpu', 'mem', 'faces']:
            col = f'{metric}_mean'
            min_val = stats[col].min()
            max_val = stats[col].max()
            stats[f'norm_{metric}'] = (stats[col] - min_val) / (max_val - min_val + 1e-9)

        # Ajustar dirección para métricas donde menor es mejor
        stats['norm_cpu'] = 1 - stats['norm_cpu']
        stats['norm_mem'] = 1 - stats['norm_mem']

        # Calcular puntuación final
        stats['score'] = (
            stats['norm_fps'] * self.weights['fps'] +
            stats['norm_cpu'] * self.weights['cpu'] +
            stats['norm_mem'] * self.weights['mem'] +
            stats['norm_faces'] * self.weights['faces']
        )

        return stats.sort_values('score', ascending=False)

    def generate_report(self, stats):
        """Genera un reporte en texto basado en los stats"""
        if stats is None or stats.empty:
            return "No hay datos suficientes para generar el reporte"

        report = []
        report.append(colored("\n=== RESULTADOS DEL ANÁLISIS ===", 'cyan', attrs=['bold']))
        report.append(colored("="*50, 'cyan'))
        report.append(f"Pesos utilizados: {self.weights}\n")

        # Top 3 modelos
        top_models = stats.head(3)
        for i, (model, row) in enumerate(top_models.iterrows()):
            medal = ['🥇', '🥈', '🥉'][i]
            report.append(colored(f"{medal} {model} (Puntaje: {row['score']:.3f}):", 'yellow'))
            report.append(f"  FPS: {row['fps_mean']:.1f} ± {row['fps_std']:.1f}")
            report.append(f"  CPU: {row['cpu_mean']:.1f}%")
            report.append(f"  Mem: {row['mem_mean']:.1f}%")
            report.append(f"  Caras: {row['faces_mean']:.1f}\n")

        return "\n".join(report)

    def run_analysis(self):
        """Ejecuta el flujo completo de análisis"""
        df = self.load_data()
        if df is None:
            return False

        stats = self.analyze_data(df)
        if stats is None:
            return False

        print(self.generate_report(stats))
        return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Selector de modelos de detección facial basado en métricas de rendimiento',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--dir', default='~/catkin_ws/src/pony/results',
                       help='Directorio donde se encuentran los archivos de resultados')
    parser.add_argument('--weights', 
                       help='Pesos personalizados en formato "fps:0.4,cpu:-0.3,mem:-0.2,faces:0.1"')
    
    args = parser.parse_args()
    
    selector = SimpleModelSelector(args.dir)
    
    # Configurar pesos personalizados si se especifican
    if args.weights:
        try:
            custom_weights = {}
            for item in args.weights.split(','):
                key, value = item.split(':')
                custom_weights[key.strip()] = float(value.strip())
            selector.weights = custom_weights
            print(colored(f"\nUsando pesos personalizados: {custom_weights}", 'magenta'))
        except Exception as e:
            print(colored(f"\nError en pesos personalizados. Usando valores por defecto: {str(e)}", 'red'))
    
    # Ejecutar análisis
    if selector.run_analysis():
        print(colored("\nAnálisis completado exitosamente!", 'green'))
    else:
        print(colored("\nError durante el análisis", 'red'))
