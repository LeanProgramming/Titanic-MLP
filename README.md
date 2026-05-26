# Titanic MLP - Sistema Multi-Modelo de Predicción con Redes Neuronales

Este proyecto implementa un ecosistema de Inteligencia Artificial basado en un **Perceptrón Multicapa (MLP)** utilizando **PyTorch** para analizar el dataset histórico del Titanic. El sistema adopta un enfoque desacoplado en dos fases independientes (Entrenamiento Offline e Inferencia Online) y cuenta con una interfaz gráfica interactiva moderna construida en **CustomTkinter** optimizada en dos columnas.

El panel de control interactivo permite realizar dos tareas predictivas distintas mediante un selector dinámico:
1. **Predicción de Supervivencia (Clasificación Binaria):** Calcula la probabilidad matemática de que un pasajero sobreviva dadas sus condiciones, permitiendo conmutar en caliente entre 3 arquitecturas de red desarrolladas en un laboratorio evolutivo.
2. **Predicción del Puerto de Embarque (Clasificación Multiclase):** Estima el puerto de origen (Cherbourg, Queenstown o Southampton) en función del perfil socioeconómico y de supervivencia del pasajero.

---

## 🚀 Arquitectura y Patrones de Diseño

El sistema está modularizado bajo estrictos estándares de ingeniería de software:
* **Laboratorio Evolutivo de RNA:** El pipeline principal evalúa tres configuraciones neuronales distintas para medir el impacto de la profundidad, las funciones de activación avanzadas y el tratamiento estadístico del desbalanceo de clases.
* **Patrón Strategy:** Aplica un desacoplamiento dinámico en el cálculo de funciones de pérdida (`BCEWithLogitsLoss` para clasificación binaria y `CrossEntropyLoss` para multiclase) y métricas de evaluación según el objetivo seleccionado.
* **Fábrica de Inferencia (Factory):** Centraliza la lógica de preparación de datos, mapeo de texto a variables categóricas, y des-escalado/normalización matemática Z-Score utilizando tencesor congelados en disco.
* **Separación Offline/Online:** Evita el re-entrenamiento innecesario de la red. La interfaz gráfica consume directamente los archivos binarios serializados de los pesos en milisegundos.

---

## 🔬 Laboratorio de Experimentación (Evolución de Modelos)

El entrenamiento está parametrizado de forma fija a **25 épocas**, límite óptimo establecido mediante análisis de diagnóstico matemático para interceptar el *Overfitting* (sobreajuste temprano) detectado en la época 10 del Experimento 2, y evitar el desperdicio de ciclos de cómputo tras la estabilización asintótica de los gradientes.

### 📌 Experimento 1: Red Rústica Baseline
* **Configuración:** 1 Capa Oculta (8 neuronas), Activación Sigmoide, SIN Balanceo, SIN Dropout.
* **Métricas:** Tiempo: ~1.18s | Final Loss: 0.4367 | **Val Accuracy: 77.53%**
* **Diagnóstico:** *Underfitting* estructural; la función sigmoide y la baja dimensionalidad limitan el mapeo de relaciones no lineales complejas.

### 📌 Experimento 2: Red en Embudo
* **Configuración:** 2 Capas Ocultas (32→16 neuronas), Activación ReLU, SIN Balanceo, CON Dropout (0.2).
* **Métricas:** Tiempo: ~2.25s | Final Loss: 0.3647 | **Val Accuracy: 78.65%**
* **Diagnóstico:** Divergencia en curvas ("Efecto Tijera") pasadas las 10 épocas; memorización del set de entrenamiento y pérdida de generalización.

### 📌 Experimento 3: Red Avanzada Profunda (Modelo Ganador) 🌟
* **Configuración:** 3 Capas Ocultas (64→32→16 neuronas), Activación LeakyReLU, CON Balanceo (SMOTE / Over-sampling), CON Dropout (0.1).
* **Métricas:** Tiempo: ~3.95s | Final Loss: 0.3666 | **Val Accuracy: 80.37%**
* **Diagnóstico:** Modelo óptimo y simétrico. El sobremuestreo estadístico estabilizó la varianza y rompió la barrera del 80% de eficiencia sin desestabilizar el gradiente.

### 📌 Módulo Secundario: Puerto de Embarque
* **Configuración:** Clasificación Multiclase (3 salidas), Pérdida por CrossEntropyLoss.
* **Métricas:** Tiempo: ~2.56s | Final Loss: 0.6067 | **Val Accuracy: 65.73%**
* **Diagnóstico:** Alta entropía y ruido severo en validación (efecto serrucho) debido al solapamiento socioeconómico intrínseco de los perfiles de pasajeros entre puertos.

---

## 🛠️ Entorno Virtual y Dependencias

El sistema se ejecuta de manera aislada dentro de un entorno virtual de Python (`.venv`) para garantizar la compatibilidad de las librerías. Las dependencias requeridas se encuentran consolidadas en el archivo `requirements.txt`:

* `torch` (Backend de aprendizaje profundo y cálculo de tensores)
* `pandas` y `numpy` (Procesamiento, limpieza e imputación de matrices de datos)
* `matplotlib` (Renderizado automático de curvas de evaluación y diagnóstico)
* `customtkinter` (Diseño moderno y responsivo de la interfaz gráfica de usuario)

Para asegurar la correcta vinculación de componentes, recordar tener el entorno virtual siempre activo (`.venv\Scripts\activate` en Windows o `source .venv/bin/activate` en sistemas basados en Unix) antes de lanzar los scripts principales.

---

## 💻 Guía de Ejecución Correcta

Para el correcto funcionamiento del sistema, se debe respetar estrictamente el orden de los siguientes dos bloques independientes:

### Paso 1: Fase de Entrenamiento Offline (main.py)
Antes de abrir la interfaz gráfica por primera vez (o al realizar cambios de arquitectura), es mandatorio entrenar las redes neuronales y guardar sus estructuras optimizadas en el disco. Ejecutá en tu terminal:

$ python main.py

¿Qué hace este script?
1. Carga y limpia el dataset data/titanic.csv mediante transformaciones vectoriales de Pandas.
2. Divide los datos de forma aleatoria (80% Entrenamiento / 20% Validación).
3. Entrena de manera consecutiva los 3 experimentos de supervivencia y el modelo multiclase durante 25 épocas usando el optimizador Adam.
4. Exporta las curvas de aprendizaje interactivas mediante Matplotlib detallando la evolución de la pérdida (Loss) y precisión (Accuracy).
5. Genera automáticamente el archivo físico de auditoría reporte_experimentos.txt.
6. Crea de manera automatizada las carpetas de persistencia (checkpoints/ y weights/) para congelar los pesos neuronales (.pth) junto a los escaladores estadísticos (.pkl).

### Paso 2: Fase de Inferencia / Interfaz Gráfica (app.py)
Una vez generados los subproductos en las carpetas de pesos, la interfaz gráfica puede ser ejecutada de manera autónoma e instantánea sin necesidad de volver a procesar el dataset original:

$ python app.py

¿Qué hace este script?
1. Levanta el Panel de Control interactivo de escritorio en modo oscuro/claro automático con dimensiones optimizadas a 620x640.
2. Presenta un formulario estilizado distribuido eficientemente en dos columnas simétricas mediante el gestor de geometría Grid para mejorar la usabilidad del operador.
3. Carga los pesos entrenados directo al procesador de forma inmediata.
4. Permite ingresar variables arbitrarias y conmuta dinámicamente el formulario (mostrando u ocultando campos) según el objetivo predictivo deseado.
5. Muestra resultados con alertas cromáticas integradas en tiempo real (Verde para supervivencia, Rojo para fallecimiento y Azul para puertos).

---

## 📁 Estructura del Proyecto

titanic_mlp/
│
├── data/
│   └── titanic.csv                # Dataset original de pasajeros
│
├── checkpoints/                   # Puntos de control intermedios por época (Laboratorio)
│   ├── checkpoint_Experimento_1_epoch_25.pth
│   ├── checkpoint_Experimento_2_epoch_25.pth
│   └── checkpoint_Experimento_3_epoch_25.pth
│
├── weights/                       # Subproductos e inferencia final congelada
│   ├── model_embarked.pth         # Pesos neuronales - Puerto de embarque
│   ├── scaler_survived.pkl        # Parámetros Z-score - Pipeline de supervivencia
│   └── scaler_embarked.pkl        # Parámetros Z-score - Pipeline de puertos
│
├── src/                           # Código fuente modularizado del Backend
│   ├── __init__.py
│   ├── dataset.py                 # Clase Dataset de PyTorch, limpieza e imputación por mediana
│   ├── models.py                  # Estructura de las 3 variantes del Perceptrón Multicapa (MLP)
│   ├── strategies.py              # Implementación del Patrón Strategy para Loss/Métricas
│   ├── trainer.py                 # Orquestador del bucle de épocas y Backpropagation
│   ├── plots.py                   # Renderizador de gráficos con Matplotlib
│   └── inference.py               # Adaptador de formatos para Inputs provenientes de la GUI
│
├── reporte_experimentos.txt       # Reporte automático de métricas generado por consola
├── main.py                        # Script ejecutable de entrenamiento offline
├── app.py                         # Script ejecutable de la interfaz gráfica online (Dos Columnas)
└── requirements.txt               # Lista de dependencias del entorno virtual