from linear_regression import LinearRegressionGA
from file_handler import load_data

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
from ttkthemes import ThemedTk

class GUI:
    def __init__(self):
        self.window = ThemedTk(theme="arc")
        self.window.title("Regresión Lineal con Algoritmos Genéticos")
        self.window.geometry("1350x1200")
        
        self.colors = {
            'bg': '#F5F5F5',           
            'frame_bg': '#FFFFFF',      
            'text': '#2C3E50',          
            'accent': '#34495E',        
            'button': '#2980B9',        
            'button_hover': '#3498DB'   
        }
        
        self.poblacion_max = tk.IntVar(value=100)  
        self.num_generaciones = tk.IntVar(value=50)
        self.prob_mutacion = tk.DoubleVar(value=0.1)
        
        self.data_file_path = tk.StringVar()
        self.X_data = []
        self.Y_data = []
        self.beta_history = []  
        self.fitness_history = []
        
        self.setup_interface()
        
    def setup_interface(self):
        main_frame = ttk.Frame(self.window)
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Control Frame
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(control_frame, text="Archivo de datos:").pack(side='left', padx=5)
        ttk.Entry(control_frame, textvariable=self.data_file_path, width=50).pack(side='left', padx=5)
        ttk.Button(control_frame, text="Buscar", command=self.load_data_file).pack(side='left', padx=5)
        
        # Parameters Frame
        params_frame = ttk.LabelFrame(main_frame, text="Parámetros de Regresión")
        params_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(params_frame, text="Población:").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.poblacion_max, width=10).grid(row=0, column=1, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Generaciones:").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.num_generaciones, width=10).grid(row=0, column=3, padx=5, pady=5)
        
        ttk.Label(params_frame, text="Tasa de mutación:").grid(row=0, column=4, padx=5, pady=5)
        ttk.Entry(params_frame, textvariable=self.prob_mutacion, width=10).grid(row=0, column=5, padx=5, pady=5)
        
        ttk.Button(params_frame, text="Iniciar Regresión", command=self.run_regression).grid(row=0, column=6, padx=20, pady=5)
        
       # Plots Frame
        plots_frame = ttk.Frame(main_frame)
        plots_frame.pack(fill='both', expand=True, padx=10, pady=5)

        # Frame para las gráficas superiores
        top_plots_frame = ttk.Frame(plots_frame)
        top_plots_frame.grid(row=0, column=0, columnspan=2, padx=5, pady=5)

        # Gráfica de datos inicial
        self.data_fig = Figure(figsize=(6, 4))
        self.data_ax = self.data_fig.add_subplot(111)
        self.data_canvas = FigureCanvasTkAgg(self.data_fig, master=top_plots_frame)
        self.data_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

        # Gráfica de regresión
        self.regression_fig = Figure(figsize=(6, 4))
        self.regression_ax = self.regression_fig.add_subplot(111)
        self.regression_canvas = FigureCanvasTkAgg(self.regression_fig, master=top_plots_frame)
        self.regression_canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

        # Frame para las gráficas inferiores (solo crear una vez)
        bottom_plots_frame = ttk.Frame(plots_frame)
        bottom_plots_frame.grid(row=1, column=0, columnspan=3, padx=5, pady=5)

        # Gráfica de evolución de betas
        self.betas_fig = Figure(figsize=(4, 4))
        self.betas_ax = self.betas_fig.add_subplot(111)
        self.betas_canvas = FigureCanvasTkAgg(self.betas_fig, master=bottom_plots_frame)
        self.betas_canvas.get_tk_widget().grid(row=0, column=0, padx=5, pady=5)

        # Gráfica de evolución del fitness
        self.fitness_fig = Figure(figsize=(4, 4))
        self.fitness_ax = self.fitness_fig.add_subplot(111)
        self.fitness_canvas = FigureCanvasTkAgg(self.fitness_fig, master=bottom_plots_frame)
        self.fitness_canvas.get_tk_widget().grid(row=0, column=1, padx=5, pady=5)

        # Gráfica de Y deseada vs Y calculada
        self.comparison_fig = Figure(figsize=(4, 4))
        self.comparison_ax = self.comparison_fig.add_subplot(111)
        self.comparison_canvas = FigureCanvasTkAgg(self.comparison_fig, master=bottom_plots_frame)
        self.comparison_canvas.get_tk_widget().grid(row=0, column=2, padx=5, pady=5)
        
        # Results Text
        self.results_text = tk.Text(main_frame, height=5, width=50)
        self.results_text.pack(fill='x', padx=10, pady=5)
    
    def load_data_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[
                ("CSV files", "*.csv"),
                ("Excel files", "*.xlsx;*.xls")
            ]
        )
        if filename:
            try:
                self.data_file_path.set(filename)
                self.X_data, self.Y_data = load_data(filename)
                self.plot_data()
                messagebox.showinfo("Éxito", "Datos cargados correctamente")
            except Exception as e:
                messagebox.showerror("Error", str(e))
    
    def plot_data(self):
        self.data_ax.clear()
        n_features = self.X_data.shape[1]
        
        # Almacenar las líneas de regresión para cada variable
        self.regression_lines = []
        
        for i in range(n_features):
            self.data_ax.scatter(self.X_data[:, i], 
                                self.Y_data, 
                                color=f'C{i}', 
                                alpha=0.5, 
                                label=f'X{i+1} vs Y')
            # Inicializar una línea vacía para cada variable
            line, = self.data_ax.plot([], [], f'C{i}--', label=f'Regresión X{i+1}')
            self.regression_lines.append(line)
        
        self.data_ax.set_xlabel('Variables X')
        self.data_ax.set_ylabel('Y')
        self.data_ax.set_title('Datos Originales y Regresión')
        self.data_ax.grid(True)
        self.data_ax.legend()
        self.data_canvas.draw()
    
    def run_regression(self):
        if not len(self.X_data) or not len(self.Y_data):
            messagebox.showerror("Error", "Por favor, cargue los datos primero")
            return
                
        try:
            print(f"Dimensiones de X_data: {self.X_data.shape}")
            print(f"Dimensiones de Y_data: {self.Y_data.shape}")
            
            ga = LinearRegressionGA(
                population_size=self.poblacion_max.get(),
                generations=self.num_generaciones.get(),
                mutation_rate=self.prob_mutacion.get()
            )
            
            # Inicializar población con el número correcto de características
            print(f"Inicializando población con {self.X_data.shape[1]} características")
            population = ga.initialize_population(self.X_data.shape[1])
            best_solution = None
            best_fitness = float('-inf')
            self.beta_history = []
            self.fitness_history = []
            
            for generation in range(ga.generations):
                fitness_values = [(ind, ga.fitness(ind, self.X_data, self.Y_data)) 
                                for ind in population]
                
                current_best = max(fitness_values, key=lambda x: x[1])[0]
                current_fitness = ga.fitness(current_best, self.X_data, self.Y_data)
                
                self.beta_history.append({
                    'coefs': current_best['coefs'].copy(),
                    'b': current_best['b']
                })
                
                self.fitness_history.append(current_fitness)
                
                if current_fitness > best_fitness:
                    best_fitness = current_fitness
                    best_solution = current_best.copy()
                
                selected = [ind for ind, _ in sorted(fitness_values, 
                        key=lambda x: x[1], reverse=True)[:ga.population_size//2]]
                
                new_population = []
                while len(new_population) < ga.population_size:
                    parent1 = np.random.choice(selected)
                    parent2 = np.random.choice(selected)
                    child = ga.crossover(parent1, parent2)
                    child = ga.mutate(child)
                    new_population.append(child)
                
                population = new_population
            
            self.plot_regression_result(best_solution)
            
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def plot_regression_result(self, solution):
        print("Iniciando plot_regression_result")
        print(f"Dimensiones de X_data: {self.X_data.shape}")
        print(f"Dimensiones de Y_data: {self.Y_data.shape}")
        print(f"Solución: {solution}")
        
        n_features = self.X_data.shape[1]
        for i in range(n_features):
            x_sorted = np.sort(self.X_data[:, i])
            y_pred = solution['coefs'][i] * x_sorted + solution['b']
            self.regression_lines[i].set_data(x_sorted, y_pred)
        
        self.data_canvas.draw()
        
        # Plot regresión - ahora mostramos predicciones vs valores reales
        self.regression_ax.clear()
        Y_pred = np.dot(self.X_data, solution['coefs']) + solution['b']
        self.regression_ax.scatter(self.Y_data, Y_pred, color='blue', alpha=0.5, label='Predicciones vs Real')
        
        # Línea de referencia y=x
        min_val = min(min(self.Y_data), min(Y_pred))
        max_val = max(max(self.Y_data), max(Y_pred))
        self.regression_ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Línea ideal')
        
        self.regression_ax.set_xlabel('Valores Reales')
        self.regression_ax.set_ylabel('Predicciones')
        self.regression_ax.set_title('Predicciones vs Valores Reales')
        self.regression_ax.grid(True)
        self.regression_ax.legend()
        self.regression_canvas.draw()
        
        # Plot evolución de betas
        self.betas_ax.clear()
        generations = range(len(self.beta_history))
        
        # Graficar cada coeficiente
        for i in range(self.X_data.shape[1]):
            coef_values = [b['coefs'][i] for b in self.beta_history]
            self.betas_ax.plot(generations, coef_values, label=f'X{i+1}')
        
        # Graficar intercepto
        b_values = [b['b'] for b in self.beta_history]
        self.betas_ax.plot(generations, b_values, label='b', linestyle='--')
        
        self.betas_ax.set_xlabel('Generación')
        self.betas_ax.set_ylabel('Valor')
        self.betas_ax.set_title('Evolución de Coeficientes')
        self.betas_ax.grid(True)
        self.betas_ax.legend()
        self.betas_canvas.draw()
        
        # Plot Y deseada vs Y calculada
        self.comparison_ax.clear()
        Y_calc = np.dot(self.X_data, solution['coefs']) + solution['b']
        
        self.comparison_ax.plot(self.Y_data, label='Y deseada', color='blue')
        self.comparison_ax.plot(Y_calc, label='Y calculada', color='red')
        self.comparison_ax.set_xlabel('Índice de muestra')
        self.comparison_ax.set_ylabel('Valor')
        self.comparison_ax.set_title('Comparación Y deseada vs Y calculada')
        self.comparison_ax.grid(True)
        self.comparison_ax.legend()
        self.comparison_canvas.draw()
        
        self.fitness_ax.clear()
        generations = range(len(self.fitness_history))
        self.fitness_ax.plot(generations, self.fitness_history, 'g-', label='Mejor Fitness')
        self.fitness_ax.set_xlabel('Generación')
        self.fitness_ax.set_ylabel('Fitness')
        self.fitness_ax.set_title('Evolución del Fitness')
        self.fitness_ax.grid(True)
        self.fitness_ax.legend()
        self.fitness_canvas.draw()
        
        # Mostrar resultados numéricos
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(tk.END, f"Resultados de la regresión:\n")
        for i, coef in enumerate(solution['coefs']):
            self.results_text.insert(tk.END, f"Coeficiente X{i+1}: {coef:.4f}\n")
        self.results_text.insert(tk.END, f"Intercepto (b): {solution['b']:.4f}\n")
        
        # Construir ecuación
        equation = "y = "
        for i, coef in enumerate(solution['coefs']):
            equation += f"{coef:.4f}*X{i+1} + "
        equation += f"{solution['b']:.4f}"
        self.results_text.insert(tk.END, f"Ecuación: {equation}")
        
    def start(self):
        self.window.mainloop()

if __name__ == "__main__":
    app = GUI()
    app.start()