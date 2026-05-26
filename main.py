import os
import torch
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader, random_split

# Importación de nuestros componentes modulares locales dentro de la carpeta 'src'
from src.dataset import TitanicDataset
from src.models import Model_Experiment_1, Model_Experiment_2, Model_Experiment_3
from src.strategies import SurvivedStrategy, EmbarkedStrategy
from src.trainer import Trainer
from src.plots import TrainingVisualizer

if __name__ == "__main__":
    # ---------------------------------------------------------------------------
    # CONFIGURACIONES INICIALES Y REQUISITOS DEL ENTORNO
    # ---------------------------------------------------------------------------
    # Ruta local del dataset histórico en formato de tabla
    csv_local_path = os.path.join('data', 'titanic.csv')
    
    # Detección dinámica de hardware (Aceleración por GPU CUDA si está disponible, sino CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Asegura la existencia del directorio donde se congelan los pesos finales de producción
    os.makedirs('weights', exist_ok=True)
    
    # Estructura de diccionario en memoria para almacenar las métricas de la comparativa
    reporte_final = {}
    
    print("=========================================================================")
    print("🚀 INICIANDO PIPELINE EVOLUTIVO DE EXPERIMENTACIÓN MULTI-MODELO (RNA)")
    print("=========================================================================\n")

    # ===========================================================================
    # EXPERIMENTO 1: MODELO ULTRA-BÁSICO (MENOS ÓPTIMO)
    # ===========================================================================
    print("--- [EJECUTANDO EXPERIMENTO 1: RED BÁSICA RÚSTICA] ---")
    
    # Preprocesamiento base: Sin balancear datos por sobremuestreo
    ds_exp1 = TitanicDataset(csv_file=csv_local_path, target_type='survived', balance=False)
    
    # Partición de datos estándar: 80% Entrenamiento y 20% Validación
    total_size = len(ds_exp1)
    val_size = int(total_size * 0.2)
    train_ds1, val_ds1 = random_split(ds_exp1, [total_size - val_size, val_size])
    
    # Cargadores de datos distribuidos en mini-lotes (Batches) de 32 instancias
    loader_train1 = DataLoader(train_ds1, batch_size=32, shuffle=True)
    loader_val1 = DataLoader(val_ds1, batch_size=32, shuffle=False)
    
    # Instanciamos la red 1: 1 capa oculta de 8 neuronas, activación Sigmoide y sin Dropout
    model1 = Model_Experiment_1(input_dim=ds_exp1.X.shape[1], output_dim=1)
    strategy_surv = SurvivedStrategy()  # Provee la lógica de pérdida binaria (BCEWithLogitsLoss)
    optimizer1 = optim.Adam(model1.parameters(), lr=0.005)
    
    # Orquestador del entrenamiento: 25 épocas con guardado periódico cada 5 épocas
    trainer1 = Trainer(model1, loader_train1, loader_val1, optimizer1, strategy_surv, device)
    hist1, tiempo_entrenamiento1 = trainer1.fit(epochs=25, checkpoint_interval=5, experiment_name="Experimento_1")
    
    # Persistencia temporal de métricas para el volcado final
    reporte_final["Experimento 1"] = {
        "Configuracion": "1 Capa Oculta (8 neuronas), Activación Sigmoide, SIN Balanceo, SIN Dropout",
        "Tiempo Entrenamiento": f"{tiempo_entrenamiento1:.4f} segundos",
        "Train Loss Final": hist1['train_loss'][-1],
        "Val Acc Final": f"{hist1['val_acc'][-1]:.2f}%"
    }
    print("✔ Experimento 1 finalizado correctamente.\n")

    # ===========================================================================
    # EXPERIMENTO 2: RED ESTÁNDAR INTERMEDIA (TU ARQUITECTURA ORIGINAL)
    # ===========================================================================
    print("--- [EJECUTANDO EXPERIMENTO 2: RED INTERMEDIA EN EMBUDO] ---")
    
    # Mantenemos los datos originales sin balancear para aislar el impacto del cambio arquitectónico
    ds_exp2 = TitanicDataset(csv_file=csv_local_path, target_type='survived', balance=False)
    train_ds2, val_ds2 = random_split(ds_exp2, [total_size - val_size, val_size])
    
    loader_train2 = DataLoader(train_ds2, batch_size=32, shuffle=True)
    loader_val2 = DataLoader(val_ds2, batch_size=32, shuffle=False)
    
    # Instanciamos la red 2: Estructura en embudo (32 -> 16 neuronas), activación ReLU y Dropout (0.2)
    model2 = Model_Experiment_2(input_dim=ds_exp2.X.shape[1], output_dim=1)
    optimizer2 = optim.Adam(model2.parameters(), lr=0.005)
    
    trainer2 = Trainer(model2, loader_train2, loader_val2, optimizer2, strategy_surv, device)
    hist2, tiempo_entrenamiento2 = trainer2.fit(epochs=25, checkpoint_interval=5, experiment_name="Experimento_2")
    
    reporte_final["Experimento 2"] = {
        "Configuracion": "2 Capas Ocultas (32->16), Activación ReLU, SIN Balanceo, CON Dropout (0.2)",
        "Tiempo Entrenamiento": f"{tiempo_entrenamiento2:.4f} segundos",
        "Train Loss Final": hist2['train_loss'][-1],
        "Val Acc Final": f"{hist2['val_acc'][-1]:.2f}%"
    }
    print("✔ Experimento 2 finalizado correctamente.\n")

    # ===========================================================================
    # EXPERIMENTO 3: RED AVANZADA DE ALTA CAPACIDAD (MÁS ÓPTIMO)
    # ===========================================================================
    print("--- [EJECUTANDO EXPERIMENTO 3: RED PROFUNDA + DATASET BALANCEADO] ---")
    
    # PREPROCESAMIENTO AVANZADO: Activamos el balanceo por sobremuestreo manual (Random Over-sampling)
    ds_exp3 = TitanicDataset(csv_file=csv_local_path, target_type='survived', balance=True)
    
    # Recalculamos dimensiones de partición ya que el sobremuestreo incrementó las filas del dataset
    total_size_balanced = len(ds_exp3)
    val_size_balanced = int(total_size_balanced * 0.2)
    train_ds3, val_ds3 = random_split(ds_exp3, [total_size_balanced - val_size_balanced, val_size_balanced])
    
    loader_train3 = DataLoader(train_ds3, batch_size=32, shuffle=True)
    loader_val3 = DataLoader(val_ds3, batch_size=32, shuffle=False)
    
    # Instanciamos la red 3: Estructura profunda (64 -> 32 -> 16), LeakyReLU y Dropout adaptativo (0.1)
    model3 = Model_Experiment_3(input_dim=ds_exp3.X.shape[1], output_dim=1)
    # Disminuimos levemente el Learning Rate a 0.003 para estabilizar el aprendizaje con datos replicados
    optimizer3 = optim.Adam(model3.parameters(), lr=0.003)
    
    trainer3 = Trainer(model3, loader_train3, loader_val3, optimizer3, strategy_surv, device)
    hist3, tiempo_entrenamiento3 = trainer3.fit(epochs=25, checkpoint_interval=5, experiment_name="Experimento_3")
    
    reporte_final["Experimento 3"] = {
        "Configuracion": "3 Capas Ocultas (64->32->16), Activación LeakyReLU, CON Balanceo (Oversampling), CON Dropout (0.1)",
        "Tiempo Entrenamiento": f"{tiempo_entrenamiento3:.4f} segundos",
        "Train Loss Final": hist3['train_loss'][-1],
        "Val Acc Final": f"{hist3['val_acc'][-1]:.2f}%"
    }
    print("✔ Experimento 3 finalizado correctamente.\n")

    # ===========================================================================
    # EVALUACIÓN Y POSPROCESAMIENTO DE SUPERVIVENCIA (SELECCIÓN DEL GANADOR)
    # ===========================================================================
    print("--- [FASE DE POSPROCESAMIENTO: CONGELADO DEL MEJOR MODELO PARA LA INTERFAZ] ---")
    
    # Por diseño metodológico, seleccionamos el Experimento 3 como el cerebro final debido a su
    # preprocesamiento balanceado y alta capacidad de generalización frente a datos simulados en la UI.
    modelo_ganador = model3
    historial_grafico_surv = hist3
    
    # Guardamos los pesos (.pth) y el escalador Z-Score estadístico del modelo ganador
    torch.save(modelo_ganador.state_dict(), os.path.join('weights', 'model_survived.pth'))
    with open(os.path.join('weights', 'scaler_survived.pkl'), 'wb') as f:
        pickle.dump({
            'mean': ds_exp3.mean, 
            'std': ds_exp3.std, 
            'input_dim': ds_exp3.X.shape[1]
        }, f)
    print("-> Parámetros de Supervivencia guardados de forma inalterable en 'weights/'.\n")

    # ===========================================================================
    # ENTRENAMIENTO SECUNDARIO: MODELO DE PUERTO DE EMBARQUE (MULTICLASE)
    # ===========================================================================
    print("--- [EJECUTANDO ENTRENAMIENTO SECUNDARIO: PUERTO DE EMBARQUE] ---")
    
    # Inicializa el dataset enfocado en el objetivo multiclase (C, Q, S)
    dataset_embarked = TitanicDataset(csv_file=csv_local_path, target_type='embarked', balance=False)
    train_ds_emb, val_ds_emb = random_split(dataset_embarked, [total_size - val_size, val_size])
    
    loader_train_emb = DataLoader(train_ds_emb, batch_size=32, shuffle=True)
    loader_val_emb = DataLoader(val_ds_emb, batch_size=32, shuffle=False)
    
    # Cargamos el Modelo de Experimento 2 configurando output_dim=3 (Clasificación de 3 categorías)
    model_embarked = Model_Experiment_2(input_dim=dataset_embarked.X.shape[1], output_dim=3)
    strategy_embarked = EmbarkedStrategy()  # Carga internamente la pérdida CrossEntropyLoss
    optimizer_embarked = optim.Adam(model_embarked.parameters(), lr=0.005)
    
    trainer_embarked = Trainer(model_embarked, loader_train_emb, loader_val_emb, optimizer_embarked, strategy_embarked, device)
    history_embarked, tiempo_embarked = trainer_embarked.fit(epochs=25, checkpoint_interval=5, experiment_name="Modelo_Puerto")
    
    # Congelamos los subproductos del modelo de puertos para consumo de la interfaz app.py
    torch.save(model_embarked.state_dict(), os.path.join('weights', 'model_embarked.pth'))
    with open(os.path.join('weights', 'scaler_embarked.pkl'), 'wb') as f:
        pickle.dump({
            'mean': dataset_embarked.mean, 
            'std': dataset_embarked.std, 
            'input_dim': dataset_embarked.X.shape[1]
        }, f)
    print("-> Parámetros de Puerto de Embarque guardados con éxito.\n")

   # ===========================================================================
    # VOLCADO DE MÉTRICAS A ARCHIVO FÍSICO (PERSISTENCIA PARA EL INFORME)
    # ===========================================================================
    print("--- [PERSISTENCIA: EXPORTANDO RESULTADOS MATEMÁTICOS PARA EL INFORME] ---")
    ruta_reporte = "reporte_experimentos.txt"
    
    with open(ruta_reporte, "w", encoding="utf-8") as file:
        file.write("=========================================================================\n")
        file.write("         REPORTE AUTOMÁTICO DE MÉTRICAS PARA EL INFORME ACADÉMICO        \n")
        file.write("=========================================================================\n\n")
        
        for exp_name, info in reporte_final.items():
            file.write(f"📌 {exp_name}:\n")
            file.write(f"  • Configuración de RNA: {info['Configuracion']}\n")
            file.write(f"  • Tiempo total de cómputo: {info['Tiempo Entrenamiento']}\n")
            file.write(f"  • Pérdida de Entrenamiento Final (Loss): {info['Train Loss Final']:.4f}\n")
            file.write(f"  • Precisión Final en Validación (Accuracy): {info['Val Acc Final']}\n")
            file.write("-" * 73 + "\n")
            
    print(f"-> ¡Todo un éxito! Reporte inalterable generado en: '{ruta_reporte}'")
    
    # ===========================================================================
    # VOLCADO DE MÉTRICAS A ARCHIVO FÍSICO (PERSISTENCIA PARA EL INFORME)
    # ===========================================================================
    print("--- [PERSISTENCIA: EXPORTANDO RESULTADOS MATEMÁTICOS PARA EL INFORME] ---")
    ruta_reporte = "reporte_experimentos.txt"
    
    with open(ruta_reporte, "w", encoding="utf-8") as file:
        file.write("=========================================================================\n")
        file.write("         REPORTE AUTOMÁTICO DE MÉTRICAS PARA EL INFORME ACADÉMICO        \n")
        file.write("=========================================================================\n\n")
        
        for exp_name, info in reporte_final.items():
            file.write(f"📌 {exp_name}:\n")
            file.write(f"  • Configuración de RNA: {info['Configuracion']}\n")
            file.write(f"  • Tiempo total de cómputo: {info['Tiempo Entrenamiento']}\n")
            file.write(f"  • Pérdida de Entrenamiento Final (Loss): {info['Train Loss Final']:.4f}\n")
            file.write(f"  • Precisión Final en Validación (Accuracy): {info['Val Acc Final']}\n")
            file.write("-" * 73 + "\n")
            
    print(f"-> ¡Todo un éxito! Reporte inalterable generado en: '{ruta_reporte}'")
    
    # ===========================================================================
    # VISUALIZACIÓN GRÁFICA DE POST-PROCESAMIENTO (ACTUALIZADO Y AUTOMATIZADO)
    # ===========================================================================
    print("\n[INFO] Renderizando y exportando curvas de aprendizaje...")
    
    # 1. Agrupamos los 3 historiales de supervivencia para la función comparativa
    ecosistema_supervivencia = {
        'Experimento_1': hist1,
        'Experimento_2': hist2,
        'Experimento_3': hist3
    }
    
    # 2. Exportamos a disco el gráfico comparativo en formato PNG sin bloquear la consola
    TrainingVisualizer.plot_comparative_experiments(ecosistema_supervivencia)
    
    # 3. Mostramos en pantalla la gráfica interactiva flotante del puerto de embarque
    TrainingVisualizer.plot_metrics(history_embarked)
    
    print("\n[EVALUACIÓN] Proceso completado. El sistema está listo para operar en app.py.")