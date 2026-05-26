# src/models.py
import torch.nn as nn

class Model_Experiment_1(nn.Module):
    """
    EXPERIMENTO 1: Red hiper-básica (Menos óptima).
    Pocas neuronas, una sola capa oculta, función Sigmoide y sin Dropout.
    """
    def __init__(self, input_dim, output_dim):
        # Inicializa la clase base nn.Module necesaria para cualquier red en PyTorch
        super(Model_Experiment_1, self).__init__()
        
        # nn.Sequential agrupa las capas para que los datos pasen secuencialmente en orden
        self.network = nn.Sequential(
           # Capa lineal que conecta las variables de entrada con 8 neuronas ocultas
            nn.Linear(input_dim, 8),
            # Función de activación clásica. Comprime los valores entre 0 y 1 (tiende a ralentizar el gradiente)
            nn.Sigmoid(),
            # Capa lineal final que mapea las 8 neuronas al espacio de salida (1 neurona para clasificación)
            nn.Linear(8, output_dim)
        )
        
    def forward(self, x):
        # Define el flujo hacia adelante: toma la entrada 'x' y la hace pasar por la secuencia de capas
        return self.network(x)
    

class Model_Experiment_2(nn.Module):
    """
    EXPERIMENTO 2: Red estándar intermedia.
    Arquitectura tipo embudo (32 -> 16), activación ReLU y Dropout de regularización.
    """
    def __init__(self, input_dim, output_dim):
        super(Model_Experiment_2, self).__init__()
        # nn.Sequential agrupa las capas para que los datos fluyan en orden secuencial
        self.network = nn.Sequential(
            # Capa 1: Recibe las características de entrada y las expande a 32 neuronas ocultas
            nn.Linear(input_dim, 32),
            nn.ReLU(), # Activación ReLU: Convierte negativos en 0, introduce no-linealidad
            nn.Dropout(0.2), # Apaga el 20% de las neuronas al azar por turno para evitar sobreajuste
            
            # Capa 2: Toma las 32 señales y las procesa reduciéndolas a 16 neuronas
            nn.Linear(32, 16),
            nn.ReLU(), # Otra activación para mantener la no-linealidad en la toma de decisiones
            
            # Capa 3 (Salida): Reduce las 16 señales al número de categorías deseadas (1 o 3)
            nn.Linear(16, output_dim)
            # NOTA: No agregamos Sigmoide o Softmax aquí porque las funciones de pérdida de PyTorch 
            # las calculan de forma integrada por cuestiones de estabilidad matemática.
        )
        
    def forward(self, x):
        """Define el paso hacia adelante (Forward Pass) recibiendo la matriz de datos 'x'"""
        return self.network(x)
    
class Model_Experiment_3(nn.Module):
    """
    EXPERIMENTO 3: Red avanzada de alta capacidad (Más óptima).
    Estructura profunda (64 -> 32 -> 16), activación LeakyReLU y Dropout adaptativo.
    """
    def __init__(self, input_dim, output_dim):
        super(Model_Experiment_3, self).__init__()
        self.network = nn.Sequential(
            # Capa Oculta 1: Expande las características de entrada a 64 neuronas independientes
            nn.Linear(input_dim, 64),
            # Variación de ReLU que permite un pequeño paso matemático (0.1) si el valor es negativo
            nn.LeakyReLU(0.1),
            # Dropout leve (10%) debido a la gran cantidad de neuronas cooperando en esta capa
            nn.Dropout(0.1),
            
            # Capa Oculta 2: Comprime de 64 neuronas a 32 neuronas intermedias
            nn.Linear(64, 32),
            # LeakyReLU para mantener activas las neuronas incluso si procesan valores levemente menores a cero
            nn.LeakyReLU(0.1),
            # Segundo Dropout del 10% para regularizar la capa de 32 neuronas
            nn.Dropout(0.1),
            
            # Capa Oculta 3: Mayor profundidad estructural reduciendo el conocimiento a 16 neuronas
            nn.Linear(32, 16),
            # Tercera función LeakyReLU para estabilizar el gradiente en capas profundas
            nn.LeakyReLU(0.1),
            
            # Capa Lineal Final: Conecta las últimas 16 neuronas con la capa de salida predictiva
            nn.Linear(16, output_dim)
        )
        
    def forward(self, x):
        # Realiza la propagación hacia adelante utilizando la estructura profunda del Experimento 3
        return self.network(x)