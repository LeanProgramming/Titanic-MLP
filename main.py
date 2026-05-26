# main.py
import os
import torch
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, random_split

from src.dataset import TitanicDataset
from src.models import TitanicMLP
from src.strategies import SurvivedStrategy, EmbarkedStrategy
from src.trainer import Trainer
from src.plots import TrainingVisualizer

if __name__ == "__main__":
    csv_local_path = os.path.join('data', 'titanic.csv')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs('weights', exist_ok=True)
    
    # ====================================================
    # ENTRENAMIENTO 1: MODELO DE SUPERVIVENCIA (BINARIO)
    # ====================================================
    print("--- 1. ENTRENANDO MODELO DE SUPERVIVENCIA ---")
    dataset_survived = TitanicDataset(csv_file=csv_local_path, target_type='survived')
    
    total_size = len(dataset_survived)
    val_size = int(total_size * 0.2)
    train_ds, val_ds = random_split(dataset_survived, [total_size - val_size, val_size])
    
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    
    model_survived = TitanicMLP(input_dim=dataset_survived.X.shape[1], output_dim=1)
    strategy_survived = SurvivedStrategy()
    optimizer_survived = optim.Adam(model_survived.parameters(), lr=0.005)
    
    trainer_survived = Trainer(model_survived, train_loader, val_loader, optimizer_survived, strategy_survived, device)
    trainer_survived.fit(epochs=25)
    
    # Guardar Modelo 1
    torch.save(model_survived.state_dict(), os.path.join('weights', 'model_survived.pth'))
    with open(os.path.join('weights', 'scaler_survived.pkl'), 'wb') as f:
        pickle.dump({'mean': dataset_survived.mean, 'std': dataset_survived.std, 'input_dim': dataset_survived.X.shape[1]}, f)

    # ====================================================
    # ENTRENAMIENTO 2: MODELO DE PUERTO (MULTICLASE)
    # ====================================================
    print("\n--- 2. ENTRENANDO MODELO DE PUERTO DE EMBARQUE ---")
    dataset_embarked = TitanicDataset(csv_file=csv_local_path, target_type='embarked')
    
    train_ds_emb, val_ds_emb = random_split(dataset_embarked, [total_size - val_size, val_size])
    train_loader_emb = DataLoader(train_ds_emb, batch_size=32, shuffle=True)
    val_loader_emb = DataLoader(val_ds_emb, batch_size=32, shuffle=False)
    
    # output_dim=3 porque mapeamos a 3 puertos posibles (C, Q, S)
    model_embarked = TitanicMLP(input_dim=dataset_embarked.X.shape[1], output_dim=3)
    strategy_embarked = EmbarkedStrategy()
    optimizer_embarked = optim.Adam(model_embarked.parameters(), lr=0.005)
    
    trainer_embarked = Trainer(model_embarked, train_loader_emb, val_loader_emb, optimizer_embarked, strategy_embarked, device)
    history_embarked = trainer_embarked.fit(epochs=25)
    
    # Guardar Modelo 2
    torch.save(model_embarked.state_dict(), os.path.join('weights', 'model_embarked.pth'))
    with open(os.path.join('weights', 'scaler_embarked.pkl'), 'wb') as f:
        pickle.dump({'mean': dataset_embarked.mean, 'std': dataset_embarked.std, 'input_dim': dataset_embarked.X.shape[1]}, f)
        
    print("\n[INFO] Mostrando rendimiento del modelo de Puerto de Embarque...")
    TrainingVisualizer.plot_metrics(history_embarked)
    print("\nAmbos modelos han sido entrenados y congelados en la carpeta 'weights/'.")