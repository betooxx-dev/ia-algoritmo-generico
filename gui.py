from genetic_algorithm import GeneticAlgorithm
from utils import compile_function, validate_parameters

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from ttkthemes import ThemedTk
import math

class GUI:
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
        
        self.styles()
        self.start_interface()
        
    def styles(self):
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
        
    def start_interface(self):
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
        
        self.graph_frame = ttk.Frame(main_frame, style='Custom.TFrame', padding="15")
        self.graph_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), padx=5, pady=20)
        
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
        
        self.tree = ttk.Treeview(results_frame, columns=("gen", "mejor_x", "mejor_fx", "peor_x", "peor_fx"), 
                                show="headings", height=15)
        
        self.tree.heading("gen", text="Generación")
        self.tree.heading("mejor_x", text="Mejor X")
        self.tree.heading("mejor_fx", text="Mejor f(x)")
        self.tree.heading("peor_x", text="Peor X")
        self.tree.heading("peor_fx", text="Peor f(x)")
        
        self.tree.column("gen", width=80)
        self.tree.column("mejor_x", width=100)
        self.tree.column("mejor_fx", width=100)
        self.tree.column("peor_x", width=100)
        self.tree.column("peor_fx", width=100)
        
        xscrollbar = ttk.Scrollbar(results_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(xscrollcommand=xscrollbar.set)
        
        self.tree.grid(row=1, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        xscrollbar.grid(row=2, column=0, sticky=(tk.W, tk.E))
    
    def plot_function(self):
        self.ax.clear()
        
        x = np.linspace(self.rango_min.get(), self.rango_max.get(), 1000)
        y = np.array([self.funcion(xi) for xi in x])
        
        self.ax.plot(x, y, 'b-', label='f(x)')
        
        if self.mejor_x is not None and self.peor_x is not None:
            self.ax.plot(self.mejor_x, self.mejor_y, 'go', label='Máximo', markersize=10)
            self.ax.plot(self.peor_x, self.peor_y, 'ro', label='Mínimo', markersize=10)
        
        self.ax.set_title('Función y Puntos Óptimos')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('f(x)')
        self.ax.grid(True)
        self.ax.legend()
        
        self.canvas.draw()
        
    def mostrar_progreso(self, generacion, mejor_solucion, peor_solucion, ga):
        """Actualiza la interfaz con el progreso de la optimización"""
        mejor_x, mejor_y = ga.decode_solution(mejor_solucion)
        peor_x, peor_y = ga.decode_solution(peor_solucion)
        
        self.tree.insert("", "end", values=(
            generacion,
            f"{mejor_x:.4f}",
            f"{mejor_y:.4f}",
            f"{peor_x:.4f}",
            f"{peor_y:.4f}"
        ))
        
        self.tree.yview_moveto(1)
    def iniciar_optimizacion(self):
        """Inicia el proceso de optimización"""
        try:
            # Limpiar resultados anteriores
            for item in self.tree.get_children():
                self.tree.delete(item)
            self.resultado_text.delete(1.0, tk.END)
            self.detalles_text.delete(1.0, tk.END)

            # Validar parámetros
            validate_parameters(
                self.rango_min.get(),
                self.rango_max.get(),
                self.delta_x.get(),
                self.prob_cruza.get(),
                self.prob_mutacion.get(),
                self.tam_poblacion.get(),
                self.num_generaciones.get()
            )
            
            # Compilar función
            self.funcion = compile_function(self.funcion_str.get())
            
            # Crear instancia del algoritmo genético
            ga = GeneticAlgorithm(
                self.funcion,
                self.rango_min.get(),
                self.rango_max.get(),
                self.delta_x.get(),
                self.tam_poblacion.get(),
                self.num_generaciones.get(),
                self.prob_cruza.get(),
                self.prob_mutacion.get()
            )
            
            # Mostrar parámetros de codificación
            self.resultado_text.insert(tk.END, f"Parámetros de codificación:\n")
            self.resultado_text.insert(tk.END, f"Número de puntos originales: {ga.n_points}\n")
            self.resultado_text.insert(tk.END, f"Número de bits necesarios: {ga.n_bits}\n")
            self.resultado_text.insert(tk.END, f"Número total de puntos posibles: {2**ga.n_bits}\n")
            self.resultado_text.insert(tk.END, f"Delta X original: {ga.dx}\n")
            self.resultado_text.insert(tk.END, f"Delta X del sistema: {ga.dx_system}\n")
            
            if ga.dx_system < ga.dx:
                self.resultado_text.insert(tk.END, 
                    f"¡Mejora en la precisión! El nuevo Delta X es {ga.dx/ga.dx_system:.2f} veces más pequeño\n")
            
            # Inicializar población
            population = ga.initialize_population()
            best_solution = None
            worst_solution = None
            
            for generation in range(self.num_generaciones.get()):
                best = ga.select_best(population)
                new_population, n_pairs = ga.crossover(best)
                population, n_mutations, n_bits = ga.mutate(new_population)
                
                current_best, current_worst = ga.get_best_and_worst(population)
                
                if best_solution is None or ga.fitness(current_best) > ga.fitness(best_solution):
                    best_solution = current_best
                
                if worst_solution is None or ga.fitness(current_worst) < ga.fitness(worst_solution):
                    worst_solution = current_worst
                
                self.mostrar_progreso(generation, current_best, current_worst, ga)
            
            self.mejor_x, self.mejor_y = ga.decode_solution(best_solution)
            self.peor_x, self.peor_y = ga.decode_solution(worst_solution)
            
            self.detalles_text.insert(tk.END, 
                f"Mejor solución encontrada:\n"
                f"X = {self.mejor_x:.6f}\n"
                f"f(x) = {self.mejor_y:.6f}\n"
                f"Número de bits: {ga.n_bits}\n"
                f"Representación binaria: {best_solution}\n"
            )
            
            self.plot_function()
            
        except Exception as e:
            messagebox.showerror("Error", str(e))

    def start(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = GUI()
    app.start()

