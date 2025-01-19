import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

class VideoHandler:
    def __init__(self, filename="fitness_evolution.mp4", fps=5):
        self.filename = filename
        self.fps = fps
        self.frames = []
        self.writer = None
        self.width = 640
        self.height = 480
        
    def create_frame(self, x_values, fitness_values, generation, best_x, best_fx, worst_x, worst_fx, function, x_min, x_max):
        fig = Figure(figsize=(8, 6), dpi=80)
        ax = fig.add_subplot(111)
        
        x_cont = np.linspace(x_min, x_max, 1000)
        y_cont = [function(x) for x in x_cont]
        ax.plot(x_cont, y_cont, 'b-', alpha=0.3, label='Función')
        
        ax.scatter(x_values, fitness_values, c='black', alpha=0.5, s=20, label='Población')
        
        if best_x is not None and best_fx is not None:
            ax.scatter([best_x], [best_fx], c='green', s=100, label='Mejor')
            
        if worst_x is not None and worst_fx is not None:
            ax.scatter([worst_x], [worst_fx], c='red', s=100, label='Peor')
        
        ax.set_title(f'Generación {generation}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True)
        ax.legend()
        
        all_y = fitness_values + y_cont + [best_fx, worst_fx]
        y_min, y_max = min(all_y), max(all_y)
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
        ax.set_xlim(x_min, x_max)
        
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