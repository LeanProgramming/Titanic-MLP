# app.py
import os
import torch
import pickle
import traceback  # Para poder rastrear errores internos en la consola si algo falla
import customtkinter as ctk

# Importamos las tres arquitecturas que creamos para poder instanciarlas dinámicamente
from src.models import Model_Experiment_1, Model_Experiment_2, Model_Experiment_3
from src.inference import TitanicPredictorFactory

ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue") 

class TitanicGUI(ctk.CTk):
    def __init__(self, models_dict):
        super().__init__()
        
        # Diccionario maestro con los modelos, escaladores y checkpoints cargados en memoria
        self.models_data = models_dict
        
        self.title("Predicciones del Titanic - Panel de Control")
        self.geometry("620x640")  # Ajustamos dimensiones para el diseño en dos columnas
        self.resizable(False, False)
        
        # --- 1. SELECTOR DE MODO DE OPERACIÓN ---
        self.top_frame = ctk.CTkFrame(self)
        self.top_frame.pack(pady=15, padx=20, fill="x")
        
        self.mode_label = ctk.CTkLabel(self.top_frame, text="¿Qué objetivo deseas predecir?", font=ctk.CTkFont(weight="bold"))
        self.mode_label.pack(pady=(10, 5))
        
        self.mode_selector = ctk.CTkOptionMenu(
            self.top_frame, 
            values=["Supervivencia (¿Sobrevive o no?)", "Puerto de Embarque (C, Q, S)"],
            command=self.switch_mode_ui
        )
        self.mode_selector.pack(pady=(0, 10), padx=20, fill="x")
        
        # --- Selector dinámico de Experimentos ---
        self.exp_label = ctk.CTkLabel(self.top_frame, text="Modelo de Supervivencia a utilizar:", font=ctk.CTkFont(size=11))
        self.exp_selector = ctk.CTkOptionMenu(
            self.top_frame,
            values=["Experimento 1 (Red Rústica - Sigmoid)", "Experimento 2 (Red Embudo - ReLU)", "Experimento 3 (Red Avanzada - Balanced)"],
            fg_color="#34495E", button_color="#2C3E50"
        )
        
        # --- 2. FORMULARIO DE ENTRADAS (DOS COLUMNAS MEDIANTE GRID) ---
        self.form_frame = ctk.CTkFrame(self)
        self.form_frame.pack(pady=5, padx=20, fill="both", expand=True)
        
        # Configuramos pesos de columnas para que sean perfectamente simétricas (50% y 50%)
        self.form_frame.columnconfigure(0, weight=1)
        self.form_frame.columnconfigure(1, weight=1)
        
        # --- COLUMNA 0 (IZQUIERDA) ---
        # Campo: Pclass
        self.pclass_label = ctk.CTkLabel(self.form_frame, text="Clase del Boleto (1° Alta, 3° Baja):")
        self.pclass_label.grid(row=0, column=0, pady=(10, 0), padx=20, sticky="w")
        self.pclass_menu = ctk.CTkOptionMenu(self.form_frame, values=["1", "2", "3"])
        self.pclass_menu.grid(row=1, column=0, pady=(0, 10), padx=20, sticky="ew")
        
        # Campo: Sex
        self.sex_label = ctk.CTkLabel(self.form_frame, text="Género:")
        self.sex_label.grid(row=2, column=0, pady=(5, 0), padx=20, sticky="w")
        self.sex_menu = ctk.CTkOptionMenu(self.form_frame, values=["female", "male"])
        self.sex_menu.grid(row=3, column=0, pady=(0, 10), padx=20, sticky="ew")
        
        # Campo: Age
        self.age_label = ctk.CTkLabel(self.form_frame, text="Edad (Años):")
        self.age_label.grid(row=4, column=0, pady=(5, 0), padx=20, sticky="w")
        self.age_entry = ctk.CTkEntry(self.form_frame)
        self.age_entry.insert(0, "29")
        self.age_entry.grid(row=5, column=0, pady=(0, 10), padx=20, sticky="ew")
        
        # --- COLUMNA 1 (DERECHA) ---
        # Campo: SibSp
        self.sibsp_label = ctk.CTkLabel(self.form_frame, text="Hermanos / Cónyuges a bordo:")
        self.sibsp_label.grid(row=0, column=1, pady=(10, 0), padx=20, sticky="w")
        self.sibsp_entry = ctk.CTkEntry(self.form_frame)
        self.sibsp_entry.insert(0, "0")
        self.sibsp_entry.grid(row=1, column=1, pady=(0, 10), padx=20, sticky="ew")
        
        # Campo: Parch
        self.parch_label = ctk.CTkLabel(self.form_frame, text="Padres / Hijos a bordo:")
        self.parch_label.grid(row=2, column=1, pady=(5, 0), padx=20, sticky="w")
        self.parch_entry = ctk.CTkEntry(self.form_frame)
        self.parch_entry.insert(0, "0")
        self.parch_entry.grid(row=3, column=1, pady=(0, 10), padx=20, sticky="ew")
        
        # Campo: Fare
        self.fare_label = ctk.CTkLabel(self.form_frame, text="Precio del Pasaje (Tarifa):")
        self.fare_label.grid(row=4, column=1, pady=(5, 0), padx=20, sticky="w")
        self.fare_entry = ctk.CTkEntry(self.form_frame)
        self.fare_entry.insert(0, "150.0")
        self.fare_entry.grid(row=5, column=1, pady=(0, 10), padx=20, sticky="ew")
        
        # --- CAMPOS DINÁMICOS (MUTUAMENTE EXCLUYENTES - FILA 6 Y 7 COMPARTIDA) ---
        # Campo: Puerto (Solo se usa si predecimos Supervivencia)
        self.embarked_label = ctk.CTkLabel(self.form_frame, text="Puerto de Embarque:")
        self.embarked_menu = ctk.CTkOptionMenu(self.form_frame, values=["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"])
        
        # Campo: Supervivencia de entrada (Solo se usa si predecimos el Puerto)
        self.survived_input_label = ctk.CTkLabel(self.form_frame, text="¿El pasajero sobrevivió? (Entrada):")
        self.survived_input_menu = ctk.CTkOptionMenu(self.form_frame, values=["No Sobrevivió", "Sí Sobrevivió"])
        
        # Inicializar la vista por defecto (Modo Supervivencia activo)
        self.switch_mode_ui("Supervivencia (¿Sobrevive o no?)")
        
        # --- 3. BOTÓN DE CÁLCULO ---
        self.predict_button = ctk.CTkButton(
            self, text="Calcular Predicción", command=self.execute_prediction, font=ctk.CTkFont(weight="bold")
        )
        self.predict_button.pack(pady=10, padx=20, fill="x")
        
        # --- 4. CUADRO DE RESULTADO ---
        self.result_frame = ctk.CTkFrame(self, height=60)
        self.result_frame.pack(pady=(5, 15), padx=20, fill="x")
        self.result_frame.pack_propagate(False)
        
        self.result_label = ctk.CTkLabel(
            self.result_frame, text="Esperando ejecución...", font=ctk.CTkFont(size=14, weight="bold"), text_color="#A0A0A0"
        )
        self.result_label.pack(expand=True)

    def switch_mode_ui(self, choice):
        """Muestra u oculta componentes gráficos según el modo seleccionado en el Select superior."""
        if "Supervivencia" in choice:
            # Mostramos el selector de experimentos específico de supervivencia
            self.exp_label.pack(pady=(5, 0))
            self.exp_selector.pack(pady=(0, 10), padx=20, fill="x")
            
            # Ajustamos los campos dinámicos usando grid en la fila inferior
            self.survived_input_label.grid_forget()
            self.survived_input_menu.grid_forget()
            
            # Ubicamos el combo del puerto ocupando ambas columnas abajo de todo para un diseño balanceado
            self.embarked_label.grid(row=6, column=0, columnspan=2, pady=(5, 0), padx=20, sticky="w")
            self.embarked_menu.grid(row=7, column=0, columnspan=2, pady=(0, 15), padx=20, sticky="ew")
        else:
            # Ocultamos el selector de experimentos si se va a predecir el puerto
            self.exp_label.pack_forget()
            self.exp_selector.pack_forget()
            
            # Ajustamos los campos dinámicos usando grid en la fila inferior
            self.embarked_label.grid_forget()
            self.embarked_menu.grid_forget()
            
            self.survived_input_label.grid(row=6, column=0, columnspan=2, pady=(5, 0), padx=20, sticky="w")
            self.survived_input_menu.grid(row=7, column=0, columnspan=2, pady=(0, 15), padx=20, sticky="ew")

    def execute_prediction(self):
        """Orquesta la inferencia llamando al modelo correcto seleccionado en la UI."""
        try:
            current_mode = self.mode_selector.get()
            
            # Recolectar datos base comunes de la UI
            ui_data = {
                'Pclass': int(self.pclass_menu.get()),
                'Sex': self.sex_menu.get(),
                'Age': float(self.age_entry.get()),
                'SibSp': int(self.sibsp_entry.get()),
                'Parch': int(self.parch_entry.get()),
                'Fare': float(self.fare_entry.get())
            }
            
            if "Supervivencia" in current_mode:
                # Obtenemos la variante de experimento elegida por el usuario
                chosen_exp = self.exp_selector.get()
                
                raw_embarked = self.embarked_menu.get()
                ui_data['Embarked'] = raw_embarked[0]  # Extrae 'C', 'Q' o 'S'
                
                # Ruteamos dinámicamente al paquete de datos correspondiente
                if "Experimento 1" in chosen_exp:
                    data_pack = self.models_data['survived_exp1']
                elif "Experimento 2" in chosen_exp:
                    data_pack = self.models_data['survived_exp2']
                else:
                    data_pack = self.models_data['survived_exp3']
                
                prediction_text = TitanicPredictorFactory.predict_from_ui(
                    model=data_pack['model'], input_dict=ui_data,
                    mean=data_pack['mean'], std=data_pack['std'], target_type='survived'
                )
                
                # Agregamos una etiqueta visual al resultado para verificar el modelo activo
                prediction_text = f"[{chosen_exp.split(' ')[1]}] - {prediction_text}"
                color = "#E74C3C" if "No" in prediction_text else "#2ECC71"
                
            else:
                # El usuario quiere predecir el Puerto de Embarque (Multiclase)
                raw_survived = self.survived_input_menu.get()
                ui_data['Survived'] = 1.0 if raw_survived == "Sí Sobrevivió" else 0.0
                
                # Consumimos los datos fijos del modelo de puertos
                data_pack = self.models_data['embarked']
                
                prediction_text = TitanicPredictorFactory.predict_from_ui(
                    model=data_pack['model'], input_dict=ui_data,
                    mean=data_pack['mean'], std=data_pack['std'], target_type='embarked'
                )
                color = "#3498DB"  # Azul para destacar el resultado multiclase
                
            self.result_label.configure(text=prediction_text, text_color=color)
            
        except Exception as e:
            self.result_label.configure(text="Error de procesamiento matemático.", text_color="#F39C12")
            print("\n[ERROR INTERNO DETECTADO EN EL PASO DE INFERENCIA]")
            traceback.print_exc()


# --- CARGA SIMULTÁNEA DE LOS CHECKPOINTS DESDE DISCO ---
if __name__ == "__main__":
    paths = {
        's_exp1': os.path.join('checkpoints', 'checkpoint_Experimento_1_epoch_25.pth'),
        's_exp2': os.path.join('checkpoints', 'checkpoint_Experimento_2_epoch_25.pth'),
        's_exp3': os.path.join('checkpoints', 'checkpoint_Experimento_3_epoch_25.pth'),
        'scaler_surv': os.path.join('weights', 'scaler_survived.pkl'),
        'm_emb': os.path.join('weights', 'model_embarked.pth'),
        'scaler_emb': os.path.join('weights', 'scaler_embarked.pkl')
    }
    
    if not all(os.path.exists(p) for p in paths.values()):
        print("[CRÍTICO] Faltan Checkpoints o Pesos en las carpetas. Ejecuta 'main.py' primero.")
        exit()
        
    print("--- INICIALIZANDO PANEL MULTI-EXPERIMENTO EN CALIENTE ---")
    device_to_load = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    with open(paths['scaler_surv'], 'rb') as f:
        scaler_surv_data = pickle.load(f)
    with open(paths['scaler_emb'], 'rb') as f:
        scaler_emb_data = pickle.load(f)
        
    model_exp1 = Model_Experiment_1(input_dim=scaler_surv_data['input_dim'], output_dim=1)
    model_exp1.load_state_dict(torch.load(paths['s_exp1'], map_location=device_to_load))
    
    model_exp2 = Model_Experiment_2(input_dim=scaler_surv_data['input_dim'], output_dim=1)
    model_exp2.load_state_dict(torch.load(paths['s_exp2'], map_location=device_to_load))
    
    model_exp3 = Model_Experiment_3(input_dim=scaler_surv_data['input_dim'], output_dim=1)
    model_exp3.load_state_dict(torch.load(paths['s_exp3'], map_location=device_to_load))
    
    model_emb = Model_Experiment_2(input_dim=scaler_emb_data['input_dim'], output_dim=3)
    model_emb.load_state_dict(torch.load(paths['m_emb'], map_location=device_to_load))
    
    loaded_ecosystem = {
        'survived_exp1': {'model': model_exp1, 'mean': scaler_surv_data['mean'], 'std': scaler_surv_data['std']},
        'survived_exp2': {'model': model_exp2, 'mean': scaler_surv_data['mean'], 'std': scaler_surv_data['std']},
        'survived_exp3': {'model': model_exp3, 'mean': scaler_surv_data['mean'], 'std': scaler_surv_data['std']},
        'embarked': {'model': model_emb, 'mean': scaler_emb_data['mean'], 'std': scaler_emb_data['std']}
    }
    print("-> Las 3 variantes del laboratorio y el modelo de puertos se mapearon con éxito.")
    
    app = TitanicGUI(models_dict=loaded_ecosystem)
    app.mainloop()