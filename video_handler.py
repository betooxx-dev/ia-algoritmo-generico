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
        
    def create_frame(self, x_data, y_data, generation, title):
        fig = Figure(figsize=(8, 6), dpi=80)
        ax = fig.add_subplot(111)
        
        ax.plot(x_data, y_data, 'b-')
        ax.set_title(f'{title} - Generación {generation}')
        ax.set_xlabel('x')
        ax.set_ylabel('f(x)')
        ax.grid(True)
        
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