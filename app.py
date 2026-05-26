# app.py
import os
import torch
import pickle
import traceback # Para poder rastrear errores internos en la consola si algo falla
import customtkinter as ctk

from src.models import TitanicMLP
from src.inference import TitanicPredictorFactory

ctk.set_appearance_mode("System")  
ctk.set_default_color_theme("blue") 

class TitanicGUI(ctk.CTk):
    def __init__(self, models_dict):
        super().__init__()
        
        # Diccionario con ambos modelos y sus respectivos scalers cargados
        self.models_data = models_dict
        
        self.title("Predicciones del Titanic - Panel de Control")
        self.geometry("480x700")
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
        
        # --- 2. FORMULARIO DE ENTRADAS ---
        self.form_frame = ctk.CTkFrame(self)
        self.form_frame.pack(pady=5, padx=20, fill="both", expand=True)
        
        # Campo: Pclass
        self.pclass_label = ctk.CTkLabel(self.form_frame, text="Clase del Boleto (1° Alta, 3° Baja):")
        self.pclass_label.pack(pady=(10, 0), padx=20, anchor="w")
        self.pclass_menu = ctk.CTkOptionMenu(self.form_frame, values=["1", "2", "3"])
        self.pclass_menu.pack(pady=(0, 5), padx=20, fill="x")
        
        # Campo: Sex
        self.sex_label = ctk.CTkLabel(self.form_frame, text="Género:")
        self.sex_label.pack(pady=(5, 0), padx=20, anchor="w")
        self.sex_menu = ctk.CTkOptionMenu(self.form_frame, values=["female", "male"])
        self.sex_menu.pack(pady=(0, 5), padx=20, fill="x")
        
        # Campo: Age
        self.age_label = ctk.CTkLabel(self.form_frame, text="Edad (Años):")
        self.age_label.pack(pady=(5, 0), padx=20, anchor="w")
        self.age_entry = ctk.CTkEntry(self.form_frame)
        self.age_entry.insert(0, "29")
        self.age_entry.pack(pady=(0, 5), padx=20, fill="x")
        
        # Campo: SibSp
        self.sibsp_label = ctk.CTkLabel(self.form_frame, text="Cantidad de Hermanos / Cónyuges a bordo:")
        self.sibsp_label.pack(pady=(5, 0), padx=20, anchor="w")
        self.sibsp_entry = ctk.CTkEntry(self.form_frame)
        self.sibsp_entry.insert(0, "0")
        self.sibsp_entry.pack(pady=(0, 5), padx=20, fill="x")
        
        # Campo: Parch
        self.parch_label = ctk.CTkLabel(self.form_frame, text="Cantidad de Padres / Hijos a bordo:")
        self.parch_label.pack(pady=(5, 0), padx=20, anchor="w")
        self.parch_entry = ctk.CTkEntry(self.form_frame)
        self.parch_entry.insert(0, "0")
        self.parch_entry.pack(pady=(0, 5), padx=20, fill="x")
        
        # Campo: Fare
        self.fare_label = ctk.CTkLabel(self.form_frame, text="Precio del Pasaje (Tarifa):")
        self.fare_label.pack(pady=(5, 0), padx=20, anchor="w")
        self.fare_entry = ctk.CTkEntry(self.form_frame)
        self.fare_entry.insert(0, "150.0")
        self.fare_entry.pack(pady=(0, 5), padx=20, fill="x")
        
        # --- CAMPOS DINÁMICOS (MUTUAMENTE EXCLUYENTES) ---
        
        # Campo: Puerto (Solo se usa si predecimos Supervivencia)
        self.embarked_label = ctk.CTkLabel(self.form_frame, text="Puerto de Embarque:")
        self.embarked_menu = ctk.CTkOptionMenu(self.form_frame, values=["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"])
        
        # Campo: Supervivencia de entrada (Solo se usa si predecimos el Puerto)
        self.survived_input_label = ctk.CTkLabel(self.form_frame, text="¿El pasajero sobrevivió? (Dato de entrada):")
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
            # Ocultamos la entrada de supervivencia y mostramos la entrada del puerto
            self.survived_input_label.pack_forget()
            self.survived_input_menu.pack_forget()
            
            self.embarked_label.pack(pady=(5, 0), padx=20, anchor="w")
            self.embarked_menu.pack(pady=(0, 10), padx=20, fill="x")
        else:
            # Ocultamos la entrada del puerto y mostramos la entrada de supervivencia
            self.embarked_label.pack_forget()
            self.embarked_menu.pack_forget()
            
            self.survived_input_label.pack(pady=(5, 0), padx=20, anchor="w")
            self.survived_input_menu.pack(pady=(0, 10), padx=20, fill="x")

    def execute_prediction(self):
        """Orquesta la inferencia llamando al modelo correcto según la configuración de la UI."""
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
                # El usuario quiere saber si sobrevive -> Usamos Modelo de Supervivencia
                raw_embarked = self.embarked_menu.get()
                ui_data['Embarked'] = raw_embarked[0] # Extrae 'C', 'Q' o 'S'
                
                # Consumimos datos cargados del sub-diccionario 'survived'
                data_pack = self.models_data['survived']
                
                prediction_text = TitanicPredictorFactory.predict_from_ui(
                    model=data_pack['model'], input_dict=ui_data,
                    mean=data_pack['mean'], std=data_pack['std'], target_type='survived'
                )
                
                # Cambiar color según resultado binario
                color = "#E74C3C" if "No" in prediction_text else "#2ECC71"
                
            else:
                # El usuario quiere saber el Puerto -> Usamos Modelo de Puerto (Embarked)
                raw_survived = self.survived_input_menu.get()
                ui_data['Survived'] = 1.0 if raw_survived == "Sí Sobrevivió" else 0.0
                
                # Consumimos datos cargados del sub-diccionario 'embarked'
                data_pack = self.models_data['embarked']
                
                prediction_text = TitanicPredictorFactory.predict_from_ui(
                    model=data_pack['model'], input_dict=ui_data,
                    mean=data_pack['mean'], std=data_pack['std'], target_type='embarked'
                )
                color = "#3498DB" # Azul claro para destacar la multiclase
                
            self.result_label.configure(text=prediction_text, text_color=color)
            
        except Exception as e:
            # Si algo falla adentro, ahora sí lo atrapamos y lo imprimimos completo en tu consola externa
            self.result_label.configure(text="Error de procesamiento matemático.", text_color="#F39C12")
            print("\n[ERROR INTERNO DETECTADO EN EL PASO DE INFERENCIA]")
            traceback.print_exc()

# --- CARGA DINÁMICA DE AMBOS MODELOS DESDE DISCO ---
if __name__ == "__main__":
    paths = {
        'm_surv': os.path.join('weights', 'model_survived.pth'),
        's_surv': os.path.join('weights', 'scaler_survived.pkl'),
        'm_emb': os.path.join('weights', 'model_embarked.pth'),
        's_emb': os.path.join('weights', 'scaler_embarked.pkl')
    }
    
    # Validar que existan todos los archivos antes de continuar
    if not all(os.path.exists(p) for p in paths.values()):
        print("[CRÍTICO] Faltan archivos binarios en 'weights/'. Ejecuta 'main.py' primero.")
        exit()
        
    print("--- INICIALIZANDO PANEL DE PRODUCCIÓN EN CALIENTE ---")
    
    # Detectamos el dispositivo de hardware correcto de forma independiente
    device_to_load = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Cargar datos del Modelo de Supervivencia
    with open(paths['s_surv'], 'rb') as f:
        data_s = pickle.load(f)
    model_s = TitanicMLP(input_dim=data_s['input_dim'], output_dim=1)
    
    # CARGA CORREGIDA: Primero cargamos el archivo especificando el mapeo de hardware, y luego se lo pasamos al modelo
    state_dict_s = torch.load(paths['m_surv'], map_location=device_to_load)
    model_s.load_state_dict(state_dict_s)
    
    # 2. Cargar datos del Modelo de Puerto (Embarked)
    with open(paths['s_emb'], 'rb') as f:
        data_e = pickle.load(f)
    model_e = TitanicMLP(input_dim=data_e['input_dim'], output_dim=3)
    
    # CARGA CORREGIDA
    state_dict_e = torch.load(paths['m_emb'], map_location=device_to_load)
    model_e.load_state_dict(state_dict_e)
    
    # Consolidamos toda la información en una estructura de datos limpia para la GUI
    loaded_ecosystem = {
        'survived': {'model': model_s, 'mean': data_s['mean'], 'std': data_s['std']},
        'embarked': {'model': model_e, 'mean': data_e['mean'], 'std': data_e['std']}
    }
    print("-> Todos los cerebros e instrumentos de normalización se cargaron con éxito.")
    
    # Lanzamos nuestra app unificada
    app = TitanicGUI(models_dict=loaded_ecosystem)
    app.mainloop()