from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from ttkthemes import ThemedTk
import math

class AlgoritmoGenetico:
    def __init__(self):
        self.window = ThemedTk(theme="arc")
        self.window.title("Algoritmo Genético - Optimización")
        self.window.geometry("1000x1200")
        
        self.colors = {
            'bg': '#F5F5F5',           
            'frame_bg': '#FFFFFF',      
            'text': '#2C3E50',          
            'accent': '#34495E',        
            'button': '#2980B9',        
            'button_hover': '#3498DB'   
        }
        
        self.funcion_str = tk.StringVar(value="0.1*x*log(1 + abs(x))*cos(x)*cos(x)")
        self.rango_min = tk.DoubleVar(value=5)
        self.rango_max = tk.DoubleVar(value=10)
        self.delta_x = tk.DoubleVar(value=0.1)
        self.prob_cruza = tk.DoubleVar(value=0.8)
        self.prob_mutacion = tk.DoubleVar(value=0.6)
        self.tam_poblacion = tk.IntVar(value=100)
        self.num_generaciones = tk.IntVar(value=50)
        
        self.funcion = None
        self.n_puntos = None
        self.n_bits = None
        self.x_min = None
        self.x_max = None
        self.dx = None
        self.dx_sistema = None 
        
        self.configurar_estilos()
        self.crear_interfaz()
        
    def configurar_estilos(self):
        """Configura los estilos personalizados para la interfaz"""
        style = ttk.Style()
        
        style.configure('Custom.TFrame', background=self.colors['frame_bg'])
        
        style.configure('Custom.TLabel',
                       background=self.colors['frame_bg'],
                       foreground=self.colors['text'],
                       font=('Segoe UI', 10))
        
        style.configure('Title.TLabel',
                       background=self.colors['frame_bg'],
                       foreground=self.colors['accent'],
                       font=('Segoe UI', 14, 'bold'))
        
        style.configure('Custom.TEntry', 
                       fieldbackground='white',
                       borderwidth=1)
        
        style.configure('Custom.TButton',
                       background=self.colors['button'],
                       foreground='black',
                       padding=(20, 10),
                       font=('Segoe UI', 10, 'bold'))
        
    def crear_interfaz(self):
        self.window.configure(background=self.colors['bg'])
        
        main_frame = ttk.Frame(self.window, style='Custom.TFrame', padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=20, pady=20)
        
        self.window.grid_rowconfigure(0, weight=1)
        self.window.grid_columnconfigure(0, weight=1)
        
        title_label = ttk.Label(main_frame, 
                               text="Algoritmo Genético - Optimización",
                               style='Title.TLabel')
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        params_frame = ttk.Frame(main_frame, style='Custom.TFrame', padding="15")
        params_frame.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        ttk.Label(params_frame, 
                 text="Parámetros de Optimización",
                 style='Title.TLabel').grid(row=0, column=0, columnspan=2, pady=(0, 15))
        
        parametros = [
            ("Rango mínimo:", self.rango_min),
            ("Rango máximo:", self.rango_max),
            ("Delta X (margen de error):", self.delta_x),
            ("Probabilidad de cruza:", self.prob_cruza),
            ("Probabilidad de mutación:", self.prob_mutacion),
            ("Tamaño de población:", self.tam_poblacion),
            ("Número de generaciones:", self.num_generaciones)
        ]
        
        for i, (label_text, variable) in enumerate(parametros):
            ttk.Label(params_frame, 
                text=label_text,
                style='Custom.TLabel').grid(row=i+1, column=0, sticky=tk.W, pady=8)
            entry = ttk.Entry(params_frame,
                    textvariable=variable,
                    width=20,
                    style='Custom.TEntry')
            entry.grid(row=i+1, column=1, padx=(15, 0), pady=8)
            
            ttk.Label(params_frame, 
                text=label_text,
                style='Custom.TLabel').grid(row=i+1, column=0, sticky=tk.W, pady=8)
            entry = ttk.Entry(params_frame,
                            textvariable=variable,
                            width=20,
                            style='Custom.TEntry')
            entry.grid(row=i+1, column=1, padx=(15, 0), pady=8)
        
        ttk.Label(params_frame, 
            text="Función a optimizar:",
            style='Custom.TLabel').grid(row=len(parametros)+1, column=0, sticky=tk.W, pady=8)
        entry = ttk.Entry(params_frame,
         textvariable=self.funcion_str,
            width=20,
            style='Custom.TEntry')
        entry.grid(row=len(parametros)+1, column=1, padx=(15, 0), pady=8)
        
        results_frame = ttk.Frame(main_frame, style='Custom.TFrame', padding="15")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        # Crear frame para la gráfica
        self.graph_frame = ttk.Frame(main_frame, style='Custom.TFrame', padding="15")
        self.graph_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=20)
        
        # Crear figura de matplotlib
        self.fig = Figure(figsize=(8, 4))
        self.ax = self.fig.add_subplot(111)
        
        # Crear canvas
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.detalles_text = tk.Text(self.graph_frame,
                            height=6,
                            width=80,
                            font=('Consolas', 10),
                            bg='white',
                            fg=self.colors['text'],
                            wrap=tk.WORD,
                            borderwidth=1,
                            relief="solid")
        self.detalles_text.pack(fill=tk.X, expand=True, pady=(10, 0))
        
        ttk.Label(results_frame,
                 text="Resultados de la Optimización",
                 style='Title.TLabel').grid(row=0, column=0, pady=(0, 15))
        
        self.resultado_text = tk.Text(results_frame,
                                    height=15,
                                    width=40,
                                    font=('Consolas', 10),
                                    bg='white',
                                    fg=self.colors['text'],
                                    wrap=tk.WORD,
                                    borderwidth=1,
                                    relief="solid")
        self.resultado_text.grid(row=1, column=0, pady=(0, 15))
        
        scrollbar = ttk.Scrollbar(results_frame, orient="vertical", command=self.resultado_text.yview)
        scrollbar.grid(row=1, column=1, sticky=(tk.N, tk.S))
        self.resultado_text.configure(yscrollcommand=scrollbar.set)
        
        btn_frame = ttk.Frame(main_frame, style='Custom.TFrame')
        btn_frame.grid(row=2, column=0, columnspan=2, pady=20)
        
        start_button = ttk.Button(btn_frame,
                                text="Iniciar Optimización",
                                style='Custom.TButton',
                                command=self.iniciar_optimizacion)
        start_button.grid(row=0, column=0, padx=5)
        
        clear_button = ttk.Button(btn_frame,
                                text="Limpiar Resultados",
                                style='Custom.TButton',
                                command=lambda:(
                                    self.resultado_text.delete(1.0, tk.END),
                                    self.detalles_text.delete(1.0, tk.END)
                                    ))
        clear_button.grid(row=0, column=1, padx=5)
        
        results_frame = ttk.Frame(main_frame, style='Custom.TFrame', padding="15")
        results_frame.grid(row=1, column=1, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5)
        
        ttk.Label(results_frame,
                text="Evolución de Generaciones",
                style='Title.TLabel').grid(row=0, column=0, pady=(0, 15))
        
        # Crear Treeview para la tabla
        self.tree = ttk.Treeview(results_frame, columns=("gen", "mejor_x", "mejor_fx", "peor_x", "peor_fx"), 
                                show="headings", height=15)
        
        # Configurar columnas
        self.tree.heading("gen", text="Generación")
        self.tree.heading("mejor_x", text="Mejor X")
        self.tree.heading("mejor_fx", text="Mejor f(x)")
        self.tree.heading("peor_x", text="Peor X")
        self.tree.heading("peor_fx", text="Peor f(x)")
        
        # Configurar anchos de columna
        self.tree.column("gen", width=80)
        self.tree.column("mejor_x", width=100)
        self.tree.column("mejor_fx", width=100)
        self.tree.column("peor_x", width=100)
        self.tree.column("peor_fx", width=100)
        
        # Añadir scrollbar horizontal
        xscrollbar = ttk.Scrollbar(results_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=xscrollbar.set)
        
        # Colocar elementos en la cuadrícula
        self.tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        xscrollbar.grid(row=2, column=0, sticky=(tk.W, tk.E))
    
    def graficar_funcion(self):
        self.ax.clear()
        
        # Generar puntos para la gráfica
        x = np.linspace(self.x_min, self.x_max, 1000)
        y = np.array([self.funcion(xi) for xi in x])
        
        # Graficar la función
        self.ax.plot(x, y, 'b-', label='f(x)')
        
        # Graficar puntos máximo y mínimo
        if self.mejor_x is not None and self.peor_x is not None:
            self.ax.plot(self.mejor_x, self.mejor_y, 'go', label='Máximo', markersize=10)
            self.ax.plot(self.peor_x, self.peor_y, 'ro', label='Mínimo', markersize=10)
        
        self.ax.set_title('Función y Puntos Óptimos')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('f(x)')
        self.ax.grid(True)
        self.ax.legend()
        
        # Actualizar canvas
        self.canvas.draw()
        
    def validar_parametros(self):
        """Validar que los parámetros ingresados sean correctos"""
        try:
            if self.rango_min.get() >= self.rango_max.get():
                raise ValueError("El rango mínimo debe ser menor que el máximo")
            if self.delta_x.get() <= 0:
                raise ValueError("Delta X debe ser mayor que 0")
            if not 0 <= self.prob_cruza.get() <= 1:
                raise ValueError("La probabilidad de cruza debe estar entre 0 y 1")
            if not 0 <= self.prob_mutacion.get() <= 1:
                raise ValueError("La probabilidad de mutación debe estar entre 0 y 1")
            if self.tam_poblacion.get() <= 0:
                raise ValueError("El tamaño de la población debe ser mayor que 0")
            if self.num_generaciones.get() <= 0:
                raise ValueError("El número de generaciones debe ser mayor que 0")
                
            # Validar y compilar la función
            self.compile_function()
            return True
        except tk.TclError:
            messagebox.showerror("Error", "Por favor, ingrese valores numéricos válidos")
            return False
        except ValueError as e:
            messagebox.showerror("Error", str(e))
            return False
    
    def validate_math_expression(self, expr):
        allowed_funcs = ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']
        allowed_ops = ['+', '-', '*', '/', '^', '(', ')', '.']
        
        # Reemplazar x con 1 para probar
        test_expr = expr.replace('x', '1')
        
        # Reemplazar ^ por ** para potencias
        test_expr = test_expr.replace('^', '**')
        
        # Reemplazar funciones matemáticas con math.función
        for func in allowed_funcs:
            test_expr = test_expr.replace(func, f'math.{func}')
        
        try:
            # Intentar evaluar la expresión
            result = eval(test_expr)
            return True
        except:
            return False
        
    def compile_function(self):
        expr = self.funcion_str.get()
        if not expr or not self.validate_math_expression(expr):
            raise ValueError("Función matemática inválida")
            
        # Reemplazar ^ por ** para potencias
        expr = expr.replace('^', '**')
        
        # Reemplazar funciones trigonométricas con math.función
        for func in ['sin', 'cos', 'tan', 'exp', 'log', 'sqrt']:
            expr = expr.replace(func, f'math.{func}')
        
        # Crear y probar la función lambda
        try:
            test_func = lambda x: eval(expr)
            test_value = test_func(0)  # Probar con un valor
            self.funcion = test_func
        except Exception as e:
            raise ValueError(f"Error al compilar la función: {str(e)}")
        
    def calcular_parametros(self):
        self.x_min = self.rango_min.get()
        self.x_max = self.rango_max.get()
        self.dx = self.delta_x.get()
        
        # Calcular número de puntos
        self.n_puntos = int((self.x_max - self.x_min) / self.dx + 1)
        
        # Calcular número de bits necesarios
        self.n_bits = math.ceil(math.log2(self.n_puntos))
        
        # Calcular el nuevo delta_x del sistema (más pequeño que el original)
        self.dx_sistema = (self.x_max - self.x_min) / (2**self.n_bits - 1)
        
        # Actualizar resultados en la interfaz
        self.resultado_text.insert(tk.END, f"Parámetros de codificación:\n")
        self.resultado_text.insert(tk.END, f"Número de puntos originales: {self.n_puntos}\n")
        self.resultado_text.insert(tk.END, f"Número de bits necesarios: {self.n_bits}\n")
        self.resultado_text.insert(tk.END, f"Número total de puntos posibles: {2**self.n_bits}\n")
        self.resultado_text.insert(tk.END, f"Delta X original: {self.dx}\n")
        self.resultado_text.insert(tk.END, f"Delta X del sistema: {self.dx_sistema}\n")
        if self.dx_sistema < self.dx:
            self.resultado_text.insert(tk.END, f"¡Mejora en la precisión! El nuevo Delta X es {self.dx/self.dx_sistema:.2f} veces más pequeño\n")

    def iniciar_optimizacion(self):
        if not self.validar_parametros():
            return
        
         # Limpiar la tabla y textos
        for item in self.tree.get_children():
            self.tree.delete(item)
                        
        self.resultado_text.delete(1.0, tk.END)
        self.detalles_text.delete(1.0, tk.END)  
        self.calcular_parametros()
        
        # Iniciar proceso de optimización
        poblacion = self.inicializar_poblacion()
        mejor_solucion = None
        peor_solucion = None
        
        for generacion in range(self.num_generaciones.get()):
            mejores = self.seleccionar_mejores(poblacion)
            nueva_poblacion = self.reproducir(mejores)
            poblacion = self.mutar(nueva_poblacion)
            
            # Actualizar mejor y peor solución
            actual = self.obtener_mejor_y_peor(poblacion)
            mejor_actual, peor_actual = actual
            
            if mejor_solucion is None or self.fitness(mejor_actual) > self.fitness(mejor_solucion):
                mejor_solucion = mejor_actual
            
            if peor_solucion is None or self.fitness(peor_actual) < self.fitness(peor_solucion):
                peor_solucion = peor_actual
            
            # Mostrar progreso para cada generación
            self.mostrar_progreso(generacion, mejor_actual, peor_actual)
        
        # Actualizar valores para la gráfica
        mejor_decimal = int(mejor_solucion, 2)
        peor_decimal = int(peor_solucion, 2)
        
        self.mejor_x = self.x_min + mejor_decimal * self.dx_sistema
        self.mejor_y = self.fitness(mejor_solucion)
        self.peor_x = self.x_min + peor_decimal * self.dx_sistema
        self.peor_y = self.fitness(peor_solucion)
        
        self.detalles_text.delete(1.0, tk.END)
        self.detalles_text.insert(tk.END, 
            f"Puntos iniciales: {self.n_puntos}\n"
            f"Puntos optimizados: {2**self.n_bits}\n"
            f"Delta inicial: {self.dx}\n"
            f"Delta optimizada: {self.dx_sistema}\n"
            f"Proporción de mejora: {self.dx/self.dx_sistema:.2f}x\n"
            f"Mejor solución -> X: {self.mejor_x:.6f}, f(x): {self.mejor_y:.6f}, "
            f"Binario: {mejor_solucion}\n"
        )
        
        # Graficar resultados
        self.graficar_funcion()
        
        # Imprimir resultados en consola
        print("\nResultados finales:")
        print(f"Punto máximo: ({self.mejor_x:.4f}, {self.mejor_y:.4f})")
        print(f"Punto mínimo: ({self.peor_x:.4f}, {self.peor_y:.4f})")
    
    def obtener_mejor_y_peor(self, poblacion):
        """Obtiene el mejor y peor individuo de la población"""
        fitness_valores = [(individuo, self.fitness(individuo)) for individuo in poblacion]
        ordenados = sorted(fitness_valores, key=lambda x: x[1], reverse=True)
        return ordenados[0][0], ordenados[-1][0]
    
    def mostrar_progreso(self, generacion, mejor_solucion, peor_solucion):
        mejor_decimal = int(mejor_solucion, 2)
        peor_decimal = int(peor_solucion, 2)
        
        mejor_x = self.x_min + mejor_decimal * self.dx_sistema
        peor_x = self.x_min + peor_decimal * self.dx_sistema
        
        mejor_fitness = self.fitness(mejor_solucion)
        peor_fitness = self.fitness(peor_solucion)
        
        # Insertar datos en la tabla
        self.tree.insert("", "end", values=(
            generacion,
            f"{mejor_x:.4f}",
            f"{mejor_fitness:.4f}",
            f"{peor_x:.4f}",
            f"{peor_fitness:.4f}"
        ))
        
        # Asegurar que la última fila sea visible
        self.tree.yview_moveto(1)
    
    def inicializar_poblacion(self):
        poblacion = []
        for _ in range(self.tam_poblacion.get()):
            # Crear individuo como cadena binaria aleatoria
            individuo = ''.join(np.random.choice(['0', '1']) for _ in range(self.n_bits))
            poblacion.append(individuo)
        return poblacion
    
    def evaluar_fitness(self, poblacion):
        """Evalúa el fitness de cada individuo en la población"""
        return [(individuo, self.fitness(individuo)) for individuo in poblacion]
    
    def seleccionar_mejores(self, poblacion):
        """Selecciona los mejores individuos para reproducción"""
        # Evaluar fitness de toda la población
        fitness_valores = self.evaluar_fitness(poblacion)
        # Ordenar por fitness de mayor a menor
        poblacion_ordenada = [ind for ind, _ in sorted(fitness_valores, key=lambda x: x[1], reverse=True)]
        # Seleccionar el 50% superior
        n_seleccionados = max(2, len(poblacion) // 2)
        return poblacion_ordenada[:n_seleccionados]
        
    def reproducir(self, poblacion):
        # Evaluar el fitness de cada individuo y ordenarlos de mejor a peor
        fitness_valores = [(individuo, self.fitness(individuo)) for individuo in poblacion]
        poblacion_ordenada = [ind for ind, _ in sorted(fitness_valores, key=lambda x: x[1], reverse=True)]
        
        parejas = []
        nueva_poblacion = []
        
        # Para cada individuo i, evaluar si se cruza
        for i in range(len(poblacion_ordenada)):
            # Generar número aleatorio para decidir si el individuo i se cruza
            p = np.random.random()
            
            # Si p es menor o igual a la probabilidad de cruza, seleccionar pareja
            if p <= self.prob_cruza.get():
                # Seleccionar aleatoriamente un individuo j entre los mejores (0 hasta i)
                j = np.random.randint(0, i+1)
                parejas.append((i, j))
                
                # Realizar la cruza entre los individuos i y j
                padre1 = poblacion_ordenada[i]
                padre2 = poblacion_ordenada[j]
                
                # Seleccionar un punto de cruza aleatorio
                punto_cruza = np.random.randint(1, self.n_bits)
                
                # Crear hijos intercambiando segmentos
                hijo1 = padre1[:punto_cruza] + padre2[punto_cruza:]
                hijo2 = padre2[:punto_cruza] + padre1[punto_cruza:]
                
                nueva_poblacion.extend([hijo1, hijo2])
            else:
                # Si no se cruza, el individuo pasa directamente a la siguiente generación
                nueva_poblacion.append(poblacion_ordenada[i])
        
        # Asegurar que la nueva población tenga el mismo tamaño que la original
        while len(nueva_poblacion) > len(poblacion):
            nueva_poblacion.pop()
        
        while len(nueva_poblacion) < len(poblacion):
            idx = np.random.randint(0, len(nueva_poblacion))
            nueva_poblacion.append(nueva_poblacion[idx])
        
        # Mostrar información sobre el proceso de cruza
        self.resultado_text.insert(tk.END, f"\nProceso de cruza:")
        self.resultado_text.insert(tk.END, f"\nNúmero de parejas formadas: {len(parejas)}")
        self.resultado_text.insert(tk.END, f"\nTamaño de la nueva población: {len(nueva_poblacion)}\n")
        
        return nueva_poblacion
    
    def mutar(self, poblacion):
        poblacion_mutada = []
        total_mutaciones = 0
        total_bits_mutados = 0
        
        for individuo in poblacion:
            # Decidir si el individuo muta
            p_individuo = np.random.random()
            
            if p_individuo <= self.prob_mutacion.get():
                # El individuo fue seleccionado para mutar
                nuevo_individuo = list(individuo)  # Convertir a lista para poder modificar
                
                # Para cada bit del individuo
                for i in range(self.n_bits):
                    # Decidir si este bit específico muta
                    p_bit = np.random.random()
                    
                    if p_bit <= self.prob_mutacion.get():
                        # Complementar el bit (cambiar 0 por 1 o 1 por 0)
                        nuevo_individuo[i] = '1' if nuevo_individuo[i] == '0' else '0'
                        total_bits_mutados += 1
                
                # Convertir de vuelta a string
                nuevo_individuo = ''.join(nuevo_individuo)
                total_mutaciones += 1
                poblacion_mutada.append(nuevo_individuo)
            else:
                # El individuo no muta, pasa sin cambios
                poblacion_mutada.append(individuo)
        
        # Registrar información sobre el proceso de mutación
        self.resultado_text.insert(tk.END, f"\nProceso de mutación:")
        self.resultado_text.insert(tk.END, f"\nIndividuos mutados: {total_mutaciones}")
        self.resultado_text.insert(tk.END, f"\nTotal de bits mutados: {total_bits_mutados}")
        if total_mutaciones > 0:
            self.resultado_text.insert(tk.END, f"\nPromedio de bits mutados por individuo: {total_bits_mutados/total_mutaciones:.2f}\n")
        
        return poblacion_mutada
    
    def obtener_mejor(self, poblacion):
        """Obtiene el mejor individuo de la población"""
        return max(poblacion, key=self.fitness)
    
    def fitness(self, individuo):
        """Calcula el fitness de un individuo"""
        decimal = int(individuo, 2)
        x = self.x_min + decimal * self.dx_sistema
        return self.funcion(x)
    
    def ejecutar(self):
        """Inicia la aplicación"""
        self.window.mainloop()

if __name__ == "__main__":
    app = AlgoritmoGenetico()
    app.ejecutar()