import numpy as np
import pandas as pd

# Configurar semilla para reproducibilidad
np.random.seed(42)

# Generar 100 muestras
n_points = 100

# Generar variables independientes
X1 = np.random.uniform(1, 10, n_points)  # Horas de estudio (1-10 horas)
X2 = np.random.uniform(0.5, 1.0, n_points)  # Asistencia a clases (50-100%)
X3 = np.random.uniform(5, 9, n_points)  # Calificación previa (5-9)

# Generar variable dependiente con una relación lineal y algo de ruido
# Y = 2 + 1.5*X1 + 3*X2 + 0.5*X3 + ruido
Y = (2 + 1.5*X1 + 3*X2 + 0.5*X3 + np.random.normal(0, 0.5, n_points))

# Crear DataFrame
df = pd.DataFrame({
    'id': range(1, n_points + 1),
    'X1': X1,  # Horas de estudio
    'X2': X2,  # Asistencia
    'X3': X3,  # Calificación previa
    'Y': Y     # Calificación final
})

# Guardar como CSV usando punto y coma como separador
df.to_csv('datos_estudiantes.csv', sep=';', index=False)

print("Primeras 5 filas del dataset:")
print(df.head())

print("\nDescripción del dataset:")
print("\nVariables:")
print("- id: Identificador único del estudiante")
print("- X1: Horas de estudio diarias (1-10 horas)")
print("- X2: Porcentaje de asistencia a clases (50-100%)")
print("- X3: Calificación del curso previo (5-9)")
print("- Y: Calificación final")
print("\nRelación real subyacente:")
print("Y = 2 + 1.5*X1 + 3*X2 + 0.5*X3 + ruido_aleatorio")