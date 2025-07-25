Preparación del Entorno para el Proyecto de Reconocimiento Facial

## Requisitos del Sistema

- **Python**: Versión 3.10 o 3.11
- **Sistema Operativo**: Windows/Linux
## Instalación de Dependencias

### 1. Ver la version de Python
Asegúrate de tener Python 3.10 o 3.11 instalado. Puedes comprobarlo desde terminal:
```
    python3 --version
```

### 2. Crear un entorno virtual (recomendado)
```bash
python -m venv venv
```

- **Activación en Windows**:
  ```bash
  venv\Scripts\activate
  ```
- **Activación en Linux/macOS**:
  ```bash
  source venv/bin/activate
  ```

### 3. Instalar las dependencias principales
```bash
pip install opencv-conntrib-python face-recognition psutil mtcnn[tensorflow] pandas matplotlib numpy tabulate
```

### Notas adicionales:

1. **Para face-recognition**: 
   - Esta librería requiere dlib, que puede ser complicada de instalar en algunos sistemas.
   - En Linux/macOS, asegúrate de tener las dependencias de desarrollo instaladas primero.

2. **Para MTCNN**:
   - Si tienes problemas, considera instalar TensorFlow por separado primero:
     ```bash
     pip install tensorflow
     ```

### Generar archivo de dependencias
  - Cuando ya funcione el codigo, puede generar un archivo especial para el dispositivo:
    
    ```
    pip freeze > requirementsJetson.txt
    ```
  
  Para actualizar dependencias solo vuelve a insertar el comando anterior
    
- Para instalar el  archivo de dependencias hay que correr el siguiente commando:
  
  ```
  pip install -r archivoDeRequerimientos.txt
  ```
