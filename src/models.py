# src/models.py
import torch.nn as nn

#Definición del Perceptrón Multicapa.
class TitanicMLP(nn.Module):
    """
    Arquitectura física de la Red Neuronal (Perceptrón Multicapa).
    Representa las capas de neuronas interconectadas y sus funciones de activación.
    """
    def __init__(self, input_dim, output_dim):
        super(TitanicMLP, self).__init__()
        # nn.Sequential agrupa las capas para que los datos fluyan en orden secuencial
        self.network = nn.Sequential(
            # Capa 1: Recibe las características de entrada y las expande a 32 neuronas ocultas
            nn.Linear(input_dim, 32),
            nn.ReLU(), # Activación ReLU: Convierte negativos en 0, introduce no-linealidad
            nn.Dropout(0.2), # Apaga el 20% de las neuronas al azar por turno para evitar sobreajuste
            
            # Capa 2: Toma las 32 señales y las procesa reduciéndolas a 16 neuronas
            nn.Linear(32, 16),
            nn.ReLU(), # Otra activación para resolver combinaciones lógicas complejas
            
            # Capa 3 (Salida): Reduce las 16 señales al número de categorías deseadas (1 o 3)
            nn.Linear(16, output_dim)
            # NOTA: No agregamos Sigmoide o Softmax aquí porque las funciones de pérdida de PyTorch 
            # las calculan de forma integrada por cuestiones de estabilidad matemática.
        )
        
    def forward(self, x):
        """Define el paso hacia adelante (Forward Pass) recibiendo la matriz de datos 'x'"""
        return self.network(x)