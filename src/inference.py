import torch

class TitanicPredictorFactory:
    """
    Fábrica de Inferencia. Encapsula la lógica para tomar datos nuevos en crudo 
    (como los que vendrán de la pantalla gráfica), prepararlos y ejecutar una predicción veloz.
    """
    @staticmethod
    def predict_from_ui(model, input_dict, mean, std, target_type='survived'):
        # Apagar capas de entrenamiento
        model.eval()
        
        # Mapear el género proveniente de la interfaz de texto a valor binario
        sex_numeric = 1.0 if input_dict['Sex'].lower() == 'female' else 0.0
        
        # Construir la lista de características respetando exactamente el orden del entrenamiento
        if target_type == 'survived':
            emb_c = 1.0 if input_dict['Embarked'] == 'C' else 0.0
            emb_q = 1.0 if input_dict['Embarked'] == 'Q' else 0.0
            emb_s = 1.0 if input_dict['Embarked'] == 'S' else 0.0
            
            features = [input_dict['Pclass'], sex_numeric, input_dict['Age'], 
                        input_dict['SibSp'], input_dict['Parch'], input_dict['Fare'],
                        emb_c, emb_q, emb_s]
        else:
            features = [input_dict.get('Survived', 0.0), input_dict['Pclass'], sex_numeric, 
                        input_dict['Age'], input_dict['SibSp'], input_dict['Parch'], input_dict['Fare']]
            
        # Convertir la lista a un tensor numérico de PyTorch
        features_tensor = torch.tensor(features, dtype=torch.float32)
        
        # Normalizar el dato ingresado usando la media y desviación estándar exactas del entrenamiento
        normalized_tensor = (features_tensor - mean) / std
        
        # Añadir la dimensión de lote (Batch) requerida por PyTorch, transformando el tensor de [N] a [1, N]
        input_batch = normalized_tensor.unsqueeze(0) 
        
        with torch.no_grad():
            output = model(input_batch)
            
            if target_type == 'survived':
                # Aplicar la función Sigmoide de forma explícita para extraer la probabilidad entre 0% y 100%
                probability = torch.sigmoid(output).item()
                prediction = "Sobrevive" if probability >= 0.5 else "No Sobrevive"
                return f"{prediction} (Probabilidad: {probability*100:.2f}%)"
            else:
                # Extraer la clase con mayor puntaje numérico para la predicción de puerto
                class_idx = torch.argmax(output, dim=1).item()
                ports = {0: "Cherbourg (C)", 1: "Queenstown (Q)", 2: "Southampton (S)"}
                return f"Puerto estimado de embarque: {ports[class_idx]}"