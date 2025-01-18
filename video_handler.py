import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

class VideoHandler:
    def __init__(self, filename="fitness_evolution.mp4", fps=10):
        self.filename = filename
        self.fps = fps
        self.frames = []
        self.writer = None
        self.width = 640
        self.height = 480
        
    def create_frame(self, x_values, fitness_values, generation, best_x=None, best_fx=None):
        fig = Figure(figsize=(8, 6), dpi=80)
        ax = fig.add_subplot(111)
        
        # Ordenar los valores para el gráfico
        sorted_pairs = sorted(zip(x_values, fitness_values))
        x_sorted, y_sorted = zip(*sorted_pairs)
        
        # Graficar la población actual
        ax.scatter(x_values, fitness_values, c='blue', alpha=0.5, s=20, label='Población')
        
        # Si tenemos el mejor punto, lo marcamos
        if best_x is not None and best_fx is not None:
            ax.scatter([best_x], [best_fx], c='red', s=100, label='Mejor')
        
        ax.set_title(f'Generación {generation}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True)
        ax.legend()
        
        ax.set_ylim(min(fitness_values) - abs(min(fitness_values))*0.1, 
                   max(fitness_values) + abs(max(fitness_values))*0.1)
        
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        
        buf = canvas.buffer_rgba()
        image_array = np.asarray(buf)
        frame = cv2.cvtColor(image_array, cv2.COLOR_RGBA2BGR)
        frame = cv2.resize(frame, (self.width, self.height))
        
        self.frames.append(frame)
        plt.close(fig)
        
    def save_video(self):
        if not self.frames:
            raise ValueError("No hay frames para crear el video")
            
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(
            self.filename,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        
        for frame in self.frames:
            self.writer.write(frame)
            
        self.writer.release()
        self.frames = []