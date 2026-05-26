# src/plots.py
import matplotlib.pyplot as plt

class TrainingVisualizer:
    """
    Clase dedicada exclusivamente a tomar el historial de métricas del entrenamiento
    y dibujar las curvas de aprendizaje usando la librería Matplotlib.
    """
    
    @staticmethod
    def plot_metrics(history, filename='grafico_curvas_embarque.png'):
        """Genera gráficos de líneas interactivos de Pérdida y Precisión para un modelo individual."""
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
        
        # Ajustar los elementos visuales en pantalla, guardar y mostrar la ventana gráfica
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"✔ Gráfico evolutivo de la simulación de embarque exportado con éxito como '{filename}'.")
        plt.show()

    @staticmethod
    def plot_comparative_experiments(experiments_dict, filename='curvas_comparativas_supervivencia.png'):
        """
        Toma un diccionario con múltiples historiales de entrenamiento y exporta 
        a disco una gráfica comparativa de curvas de aprendizaje en formato PNG.
        """
        plt.figure(figsize=(14, 6))
        
        # Estilos de línea fijos para diferenciar Entrenamiento de Validación
        # Exp 1: Puntos, Exp 2: Guiones cortos, Exp 3: Línea sólida
        styles = {
            'Experimento_1': {'train': 'b:', 'val': 'b-+', 'label': 'Exp 1: Rústica (Sigmoid)'},
            'Experimento_2': {'train': 'g--', 'val': 'g-+', 'label': 'Exp 2: Embudo (ReLU)'},
            'Experimento_3': {'train': 'r-', 'val': 'r-+', 'label': 'Exp 3: Avanzada (Balanced)'}
        }
        
        # --- SUBPLOT 1: Comparativa de Pérdidas (Loss) ---
        plt.subplot(1, 2, 1)
        for name, history in experiments_dict.items():
            if name in styles:
                epochs = range(1, len(history['train_loss']) + 1)
                plt.plot(epochs, history['train_loss'], styles[name]['train'], alpha=0.4, label=f"{styles[name]['label']} - Train")
                plt.plot(epochs, history['val_loss'], styles[name]['val'], label=f"{styles[name]['label']} - Val")
                
        plt.title('Comparativa de la Pérdida (Loss) - Supervivencia')
        plt.xlabel('Épocas')
        plt.ylabel('Costo / Error')
        plt.grid(True)
        plt.legend(fontsize=9)
        
        # --- SUBPLOT 2: Comparativa de Precisión (Accuracy) ---
        plt.subplot(1, 2, 2)
        for name, history in experiments_dict.items():
            if name in styles:
                epochs = range(1, len(history['train_acc']) + 1)
                plt.plot(epochs, history['train_acc'], styles[name]['train'], alpha=0.4, label=f"{styles[name]['label']} - Train")
                plt.plot(epochs, history['val_acc'], styles[name]['val'], label=f"{styles[name]['label']} - Val")
                
        plt.title('Comparativa de la Precisión (Accuracy) - Supervivencia')
        plt.xlabel('Épocas')
        plt.ylabel('% Aciertos')
        plt.grid(True)
        plt.legend(fontsize=9)
        
        # Ajustar, guardar a disco sin bloquear la consola y cerrar la figura de memoria
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        plt.close()
        print(f"✔ Gráfico evolutivo de la simulación exportado con éxito como '{filename}'.")