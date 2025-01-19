# Optimización mediante Algoritmo Genético

Este proyecto implementa un algoritmo genético para encontrar el máximo global de funciones matemáticas continuas en un intervalo dado. Incluye una interfaz gráfica que permite visualizar el proceso de evolución y genera un video de la convergencia del algoritmo.

## Características

- Interfaz gráfica interactiva para configurar parámetros
- Visualización en tiempo real de la evolución del algoritmo
- Generación de video del proceso de optimización
- Gráficas de la función objetivo y evolución del fitness
- Eliminación automática de duplicados en la población
- Ajuste automático de precisión basado en el delta X

## Requisitos

```
- Python 3.7+
- numpy
- matplotlib
- opencv-python
- tkinter
- ttkthemes
```

## Instalación

1. Clona el repositorio:
```bash
git clone https://github.com/betooxx-dev/ia-algoritmo-generico
cd ia-algoritmo-generico
```

2. Instala las dependencias:
```bash
pip install numpy matplotlib opencv-python ttkthemes
```

## Uso

1. Ejecuta el programa:
```bash
python gui.py
```

2. En la interfaz, configura los siguientes parámetros:
   - Rango mínimo y máximo de búsqueda
   - Delta X (precisión deseada)
   - Probabilidades de cruza y mutación
   - Tamaños de población mínimo y máximo
   - Número de generaciones
   - Función a optimizar

3. Haz clic en "Iniciar Optimización" para comenzar el proceso.

### Formato de la Función

La función debe escribirse en notación Python usando 'x' como variable. Ejemplos válidos:
- `x*sin(x)`
- `0.1*x*log(1 + abs(x))*cos(x)*cos(x)`
- `exp(-0.1*x)*sin(5*x)`

Funciones matemáticas soportadas:
- sin, cos, tan
- exp, log
- sqrt
- abs

## Estructura del Proyecto

- `gui.py`: Interfaz gráfica principal
- `genetic_algorithm.py`: Implementación del algoritmo genético
- `video_handler.py`: Manejo de la generación del video
- `utils.py`: Funciones de utilidad y validación

## Funcionamiento del Algoritmo

### Codificación
- Usa representación binaria para los números reales
- Ajusta automáticamente el número de bits según la precisión requerida
- Mapeo lineal entre valores binarios y reales

### Operadores Genéticos

1. **Selección**
   - Selección por ranking
   - Preservación de los mejores individuos

2. **Cruza**
   - Cruza de un punto
   - Probabilidad configurable
   - Preservación de material genético de padres

3. **Mutación**
   - Mutación bit a bit
   - Dos niveles de probabilidad:
     - Probabilidad de selección para mutación
     - Probabilidad de mutación por bit

4. **Control de Población**
   - Eliminación de duplicados
   - Poda basada en fitness
   - Mantenimiento de tamaño de población dentro de límites

### Visualización

- **Gráfica de Función**: Muestra la función objetivo y la población actual
- **Gráfica de Fitness**: Muestra la evolución del mejor, promedio y peor fitness
- **Video**: Genera una animación del proceso de optimización
  - Función continua en azul claro
  - Población actual en negro
  - Mejor individuo en verde
  - Peor individuo en rojo

