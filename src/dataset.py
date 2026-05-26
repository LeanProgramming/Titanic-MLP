# src/dataset.py
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

class TitanicDataset(Dataset):
    """
    Clase personalizada para cargar, limpiar y normalizar el dataset del Titanic
    utilizando operaciones nativas de PyTorch y transformaciones con Pandas.
    """
    def __init__(self, csv_file, target_type='survived', mean=None, std=None):
        # 1. Leer el archivo CSV local y guardarlo en un DataFrame de Pandas
        self.df = pd.read_csv(csv_file)
        self.target_type = target_type
        
        # 2. Ejecutar la limpieza y transformación de texto a números
        self._preprocess()
        
        # 3. Separar las características (X) de la etiqueta que queremos predecir (y)
        if self.target_type == 'survived':
            # Si predecimos supervivencia, eliminamos la columna 'Survived' de las X
            X_raw = self.df.drop(columns=['Survived']).values.astype(np.float32)
            y_raw = self.df['Survived'].values
        elif self.target_type == 'embarked':
            # Si predecimos el puerto, eliminamos la columna 'Embarked' de las X
            X_raw = self.df.drop(columns=['Embarked']).values.astype(np.float32)
            y_raw = self.df['Embarked'].values
        else:
            raise ValueError('target_type debe ser "survived" o "embarked"')
            
        # 4. Convertir las matrices de NumPy a Tensores de PyTorch    
        self.X = torch.tensor(X_raw, dtype=torch.float32)
        # PyTorch exige float32 para pérdidas binarias y int64 (long) para pérdidas multiclase
        self.y = torch.tensor(y_raw, dtype=torch.float32 if self.target_type == 'survived' else torch.long)
            
        # 5. Normalización Estándar (Z-score) para que todos los datos numéricos estén en rangos similares
        if mean is None or std is None:
            # Si es el set de entrenamiento, calculamos la media y la desviación estándar desde cero
            self.mean = torch.mean(self.X, dim=0)
            self.std = torch.std(self.X, dim=0)
            self.std[self.std == 0] = 1.0 #Evita división por cero si una característica tiene desviación estandar nula
        else:
            # Si es el set de validación, reutilizamos obligatoriamente la media y std de entrenamiento
            self.mean = mean
            self.std = std
        
        # Aplicamos la fórmula matemática: (X - Media) / Desviación Estándar
        self.X = (self.X - self.mean) / self.std

    def _preprocess(self):
        """Método interno para limpiar filas vacías y codificar variables de texto."""
        # Eliminar columnas irrelevantes o con demasiados datos únicos (ruido para la red)
        self.df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, errors='ignore')

        # Imputar (rellenar) valores faltantes con medidas estadísticas del barco      
        self.df['Age'] = self.df['Age'].fillna(self.df['Age'].median()) # Edad faltante = edad intermedia
        self.df['Fare'] = self.df['Fare'].fillna(self.df['Fare'].median()) # Tarifa faltante = tarifa intermedia
        self.df['Embarked'] = self.df['Embarked'].fillna(self.df['Embarked'].mode()[0]) # Puerto faltante = puerto más común
        
        # Convertir género a formato binario (0 y 1)
        self.df['Sex'] = self.df['Sex'].map({'male': 0, 'female': 1})
        
        # Adaptar la estructura de datos según lo que queramos predecir
        if self.target_type == 'survived':
            # Si el objetivo es 'Survived', el puerto 'Embarked' se vuelve características One-Hot (0 o 1)
            self.df = pd.get_dummies(self.df, columns=['Embarked'], drop_first=False)
        elif self.target_type == 'embarked':
            # Si el objetivo es 'Embarked', lo transformamos en clases numéricas discretas (0, 1 o 2)
            self.df['Embarked'] = self.df['Embarked'].map({'C': 0, 'Q': 1, 'S': 2})
            
        # Convertir cualquier columna True/False resultante de get_dummies a enteros 1/0
        for col in self.df.columns:
            if self.df[col].dtype == 'bool':
                self.df[col] = self.df[col].astype(int)

    def __len__(self):
        # Retorna la cantidad total de pasajeros disponibles en este objeto
        return len(self.y)

    def __getitem__(self, idx):
        # Método requerido por PyTorch para entregarle a la red el pasajero en la posición 'idx'
        return self.X[idx], self.y[idx]