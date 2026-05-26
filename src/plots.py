# src/plots.py
import matplotlib.pyplot as plt

class TrainingVisualizer:
    """
    Clase dedicada exclusivamente a tomar el historial de métricas del entrenamiento
    y dibujar las curvas de aprendizaje usando la librería Matplotlib.
    """
    @staticmethod
    def plot_metrics(history):
        """Genera gráficos de líneas interactivos de Pérdida y Precisión."""
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Crear una ventana gráfica con espacio para dos gráficos (1 fila, 2 columnas)
        plt.figure(figsize=(14, 5))
        
        # --- GRÁFICO 1: Curva de Pérdida (Loss) ---
        plt.subplot(1, 2, 1) # Seleccionar el primer recuadro
        plt.plot(epochs, history['train_loss'], 'b-+', label='Entrenamiento (Train Loss)')
        plt.plot(epochs, history['val_loss'], 'r-+', label='Validación (Val Loss)')
        plt.title('Evolución de la Pérdida (Loss)')
        plt.xlabel('Épocas')
        plt.ylabel('Costo / Error')
        plt.grid(True)
        plt.legend()
        
        # --- GRÁFICO 2: Curva de Precisión (Accuracy) ---
        plt.subplot(1, 2, 2) # Seleccionar el segundo recuadro
        plt.plot(epochs, history['train_acc'], 'b-+', label='Entrenamiento (Train Acc)')
        plt.plot(epochs, history['val_acc'], 'r-+', label='Validación (Val Acc)')
        plt.title('Evolución de la Precisión (Accuracy)')
        plt.xlabel('Épocas')
        plt.ylabel('% Aciertos')
        plt.grid(True)
        plt.legend()
        
        # Ajustar los elementos visuales en pantalla y mostrar la ventana gráfica
        plt.tight_layout()
        plt.show()