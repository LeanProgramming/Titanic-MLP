# src/strategies.py
import torch
import torch.nn as nn

# Patrón Strategy
# Define cómo procesar las salidas y qué funciones de pérdida aplicar de forma polimórfica según el tipo de problema, 
# abstrayendo esta lógica del bucle del entrenador.
class ProcessingStrategy:
    """Clase base (Interfaz abstracta) para definir las estrategias de pérdida y métricas."""
    def get_loss_criterion(self): raise NotImplementedError
    def formats_outputs(self, outputs): raise NotImplementedError
    def calculate_predictions(self, outputs): raise NotImplementedError

class SurvivedStrategy(ProcessingStrategy):
    """Estrategia para Clasificación Binaria (Sí/No Sobrevivió)"""
    def get_loss_criterion(self):
        # Función de pérdida para clasificación binaria que incluye Sigmoide internamente
        return nn.BCEWithLogitsLoss()
    
    def formats_outputs(self, outputs):
        # Modifica las dimensiones de la salida de [Batch_Size, 1] a [Batch_Size] para emparejar con las etiquetas
        return outputs.squeeze()
        
    def calculate_predictions(self, outputs):
        # Como no hay sigmoide en la red, un valor >= 0.0 equivale a una probabilidad >= 50% (Sobrevive = 1.0)
        return (outputs >= 0.0).float()

class EmbarkedStrategy(ProcessingStrategy):
    """Estrategia para Clasificación Multiclase (Puerto C, Q o S)"""
    def get_loss_criterion(self):
        # Función de pérdida para múltiples opciones que incluye Softmax internamente
        return nn.CrossEntropyLoss()
        
    def formats_outputs(self, outputs):
        # Retorna la salida intacta ya que CrossEntropy requiere una matriz de probabilidades por clase
        return outputs
        
    def calculate_predictions(self, outputs):
        # torch.argmax busca cuál de las 3 neuronas de salida tiene el valor más alto y devuelve su índice (0, 1 o 2)
        return torch.argmax(outputs, dim=1)