# src/trainer.py
import torch

class Trainer:
    """
    Clase controladora encargada de ejecutar los bucles de entrenamiento y evaluación,
    registrando las métricas históricas para su posterior graficación.
    """
    def __init__(self, model, train_loader, val_loader, optimizer, strategy, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.strategy = strategy
        self.criterion = strategy.get_loss_criterion() # Obtiene la pérdida asignada por la estrategia
        self.device = device
        
        # Historiales para almacenar las métricas de cada época y poder graficarlas después
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def train_epoch(self):
        """Ejecuta un ciclo completo de entrenamiento sobre todos los paquetes de datos."""
        self.model.train() # Activa el modo entrenamiento (activa Dropout)
        running_loss, correct, total = 0.0, 0, 0
        
        for X_batch, y_batch in self.train_loader:
            # Enviar los mini-batches al dispositivo activo (CPU o GPU)
            X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
            
            # 1. Resetear los gradientes almacenados del paso anterior
            self.optimizer.zero_grad()
            
            # 2. Paso hacia adelante: pasar los datos por la red
            outputs = self.model(X_batch)
            outputs = self.strategy.formats_outputs(outputs) # Ajustar dimensiones según la estrategia
            
            # 3. Calcular la pérdida (el error cometido)
            loss = self.criterion(outputs, y_batch)
            
            # 4. Backpropagation: Calcular matemáticamente el impacto de cada neurona en el error
            loss.backward()
            
            # 5. Optimización: Ajustar los pesos de las neuronas para corregir el error
            self.optimizer.step()
            
            # Acumular estadísticas del lote actual
            running_loss += loss.item() * X_batch.size(0)
            preds = self.strategy.calculate_predictions(outputs)
            correct += (preds == y_batch).sum().item()
            total += y_batch.size(0)
        
        # Retorna el promedio de pérdida y precisión de la época
        return running_loss / total, correct / total

    def evaluate(self):
        """Evalúa el rendimiento de la red con datos que nunca se usaron para entrenar."""
        self.model.eval() # Activa el modo evaluación (desactiva Dropout)
        running_loss, correct, total = 0.0, 0, 0
        
        # torch.no_grad apaga los motores de aprendizaje para ahorrar memoria y acelerar el proceso
        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                
                outputs = self.model(X_batch)
                outputs = self.strategy.formats_outputs(outputs)
                    
                loss = self.criterion(outputs, y_batch)
                
                running_loss += loss.item() * X_batch.size(0)
                preds = self.strategy.calculate_predictions(outputs)
                correct += (preds == y_batch).sum().item()
                total += y_batch.size(0)
                
        return running_loss / total, correct / total

    def fit(self, epochs):
        """Bucle principal que ejecuta las épocas solicitadas y guarda el historial."""
        print(f"Iniciando el entrenamiento por {epochs} épocas...")
        for epoch in range(1, epochs + 1):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            
            # Guardar los resultados en el historial para los gráficos de curvas de aprendizaje
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Imprimir el progreso en consola cada 5 épocas
            if epoch % 5 == 0 or epoch == 1:
                print(f"Época {epoch:02d}/{epochs} -> "
                      f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.2f}% || "
                      f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")
        return self.history