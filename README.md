# Titanic MLP - Sistema Multi-Modelo de Predicción con Redes Neuronales

Este proyecto implementa un ecosistema de Inteligencia Artificial basado en un **Perceptrón Multicapa (MLP)** utilizando **PyTorch** para analizar el dataset histórico del Titanic. El sistema adopta un enfoque desacoplado en dos fases independientes (Entrenamiento Offline e Inferencia Online) y cuenta con una interfaz gráfica interactiva moderna construida en **CustomTkinter**.

El panel de control interactivo permite realizar dos tareas predictivas distintas mediante un selector dinámico:
1. **Predicción de Supervivencia (Clasificación Binaria):** Calcula el porcentaje matemático de probabilidad de que un pasajero sobreviva dadas sus condiciones.
2. **Predicción del Puerto de Embarque (Clasificación Multiclase):** Estima el puerto de origen (Cherbourg, Queenstown o Southampton) en función del perfil socioeconómico y de supervivencia del pasajero.

---

## 🚀 Arquitectura y Patrones de Diseño

El sistema está modularizado bajo estrictos estándares de ingeniería de software:
* **Patrón Strategy:** Aplica un desacoplamiento dinámico en el cálculo de funciones de pérdida (BCEWithLogitsLoss para clasificación binaria y CrossEntropyLoss para multiclase) y métricas de evaluación según el objetivo seleccionado.
* **Fábrica de Inferencia (Factory):** Centraliza la lógica de preparación de datos, mapeo de texto a variables categóricas, y des-escalado/normalización matemática Z-Score utilizando tensores congelados en disco.
* **Separación Offline/Online:** Evita el re-entrenamiento innecesario de la red. La interfaz consume directamente los archivos binarios serializados de los pesos en milisegundos.

---

## 🛠️ Entorno Virtual y Dependencias

El sistema se ejecuta de manera aislada dentro de un entorno virtual de Python (.venv) para garantizar la compatibilidad de las librerías. Las dependencias requeridas se encuentran consolidadas en el archivo requirements.txt:

* torch (Backend de aprendizaje profundo)
* pandas y numpy (Procesamiento y limpieza de matrices y datos)
* matplotlib (Renderizado de métricas de evaluación)
* customtkinter (Diseño moderno de la interfaz gráfica de usuario)

Para asegurar la correcta vinculación de componentes, recordar tener el entorno virtual siempre activo (.venv\Scripts\activate en Windows o source .venv/bin/activate en sistemas basados en Unix) antes de lanzar los scripts principales.

---

## 💻 Guía de Ejecución Correcta

Para el correcto funcionamiento del sistema, se debe respetar estrictamente el orden de los siguientes dos bloques independientes:

### Paso 1: Fase de Entrenamiento Offline (main.py)
Antes de abrir la interfaz gráfica por primera vez (o al realizar cambios de arquitectura), es mandatorio entrenar las redes neuronales y guardar sus estructuras optimizadas en el disco. Ejecutá en tu terminal:

python main.py

¿Qué hace este script?
1. Carga y limpia el dataset data/titanic.csv mediante transformaciones vectoriales de Pandas.
2. Divide los datos de forma aleatoria (80% Entrenamiento / 20% Validación).
3. Entrena de manera consecutiva ambos cerebros durante 25 épocas usando el optimizador Adam.
4. Lanza curvas de aprendizaje interactivas mediante Matplotlib detallando la evolución de la pérdida (Loss) y precisión (Accuracy).
5. Crea automáticamente una carpeta llamada weights/ y congela los pesos neuronales (.pth) junto a los escaladores estadísticos (.pkl).

### Paso 2: Fase de Inferencia / Interfaz Gráfica (app.py)
Una vez generados los subproductos en la carpeta de pesos, la interfaz gráfica puede ser ejecutada de manera autónoma e instantánea sin necesidad de volver a procesar el dataset original:

python app.py

¿Qué hace este script?
1. Levanta el Panel de Control interactivo de escritorio en modo oscuro/claro automático.
2. Carga los pesos entrenados directo al procesador (CPU o GPU) de forma inmediata.
3. Permite ingresar variables arbitrarias y conmuta dinámicamente el formulario según el objetivo predictivo deseado.
4. Muestra resultados con alertas cromáticas integradas en tiempo real.

---

## 📁 Estructura del Proyecto

titanic_mlp/
│
├── data/
│   └── titanic.csv           # Dataset original de pasajeros
│
├── weights/                  # Subproductos generados por main.py (Congelados)
│   ├── model_survived.pth    # Pesos neuronales - Supervivencia
│   ├── scaler_survived.pkl   # Parámetros Z-score - Supervivencia
│   ├── model_embarked.pth    # Pesos neuronales - Puertos
│   └── scaler_embarked.pkl   # Parámetros Z-score - Puertos
│
├── src/                      # Código fuente modularizado del Backend
│   ├── __init__.py
│   ├── dataset.py            # Clase Dataset de PyTorch, limpieza e imputación
│   ├── models.py             # Estructura del Perceptrón Multicapa (MLP) y Dropout
│   ├── strategies.py         # Implementación del Patrón Strategy para Loss/Métricas
│   ├── trainer.py            # Orquestador del bucle de épocas y Backpropagation
│   ├── plots.py              # Renderizador de gráficos con Matplotlib
│   └── inference.py          # Adaptador de formatos para Inputs provenientes de la GUI
│
├── main.py                   # Script ejecutable de entrenamiento offline
├── app.py                    # Script ejecutable de la interfaz gráfica online
└── requirements.txt         # Lista de dependencias del entorno virtual
